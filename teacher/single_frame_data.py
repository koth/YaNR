"""Manifest-backed single-frame sampling; independent of old FrameSource.

Build once with prepare_single_data.py. Splits are by source image/video group,
with byte-identical images joining groups before deduplication. Sampling is
source -> scene -> frame, not uniform over every extracted video frame.
All ROI coordinates refer to the EXIF-oriented image, in original pixels.
"""
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from nr_geometry import geometry_from_valid


EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
MODES = ('roi_native', 'random_native', 'scaled_crop', 'context')
RESAMPLE = getattr(Image, 'Resampling', Image).LANCZOS


def digest_file(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', str(path))]


def build_manifest(cfg, data_root=None, roi_path=None, detect_faces=False, groups_path=None):
    root = Path(data_root or cfg['data_root']).resolve()
    started = time.perf_counter()
    roi_map, group_map = {}, {}
    if roi_path:
        with open(roi_path, encoding='utf-8') as handle:
            for line in handle:
                item = json.loads(line)
                path = Path(item['path'])
                roi_map[str((path if path.is_absolute() else root / path).resolve())] = item['rois']
    if groups_path:
        with open(groups_path, encoding='utf-8') as handle:
            for line in handle:
                item = json.loads(line)
                path = Path(item['path'])
                group = str(item['group']).strip()
                if not group:
                    raise ValueError('explicit group must not be empty')
                group_map[str((path if path.is_absolute() else root / path).resolve())] = group
    detector = None
    if detect_faces:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('face indexing needs opencv-python-headless, or supply --roi-jsonl') from exc
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if detector.empty():
            raise RuntimeError('OpenCV frontal-face cascade is unavailable; supply --roi-jsonl')

    entries, rejected = [], []
    excluded = set(cfg.get('exclude_components', ['proxy_proc']))
    for source in cfg['sources']:
        source_root = root / source['directory']
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        if source.get('group_by', 'file') not in ('file', 'directory'):
            raise ValueError('group_by must be explicitly file or directory')
        paths = sorted((p for p in source_root.rglob('*') if p.suffix.lower() in EXTENSIONS
                        and not excluded.intersection(p.parts)), key=natural_key)
        frame_counts = {}
        scanned_source = 0
        for path in paths:
            relative = path.relative_to(source_root)
            explicit_group = group_map.get(str(path.resolve()))
            group = explicit_group or (relative.as_posix() if source.get('group_by', 'file') == 'file'
                                       else relative.parent.as_posix())
            frame_index = frame_counts.get(group, 0)
            frame_counts[group] = frame_index + 1
            if frame_index % int(source.get('frame_stride', 1)):
                continue
            # 进度打印:整轮要跑几十万次解码 + Haar 检测,以前只在结束时出一行统计,
            # 卡住还是慢根本分不清。每 500 张打一次,带已耗时和 ROI 命中数。
            scanned_source += 1
            if scanned_source % 500 == 0:
                elapsed = time.perf_counter() - started
                rate = scanned_source / elapsed if elapsed > 0 else 0.0
                print(f'  [{source["name"]}] {scanned_source}/{len(paths)} '
                      f'total={len(entries)} {elapsed:.0f}s {rate:.1f}/s', flush=True)
            try:
                image = read_rgb(path)
                w, h = image.size
                if min(w, h) < 33:
                    raise ValueError('image smaller than teacher minimum valid dimension 33')
                if max(w / h, h / w) > float(cfg.get('max_aspect_ratio', 8)):
                    raise ValueError('extreme aspect ratio exceeds configured coverage')
                rois = roi_map.get(str(path.resolve()), [])
                if detector is not None and not rois:
                    scale = min(1.0, 1024.0 / max(w, h))
                    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), RESAMPLE)
                    gray = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2GRAY)
                    rois = [[int(x / scale), int(y / scale),
                             min(w, int((x + rw) / scale)), min(h, int((y + rh) / scale))]
                            for x, y, rw, rh in detector.detectMultiScale(
                                gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))]
                for box in rois:
                    if len(box) != 4 or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                        raise ValueError('invalid ROI in EXIF-oriented original-pixel coordinates')
                entries.append(dict(path=str(path.resolve()), source=source['name'],
                                    group=('explicit/' + group if explicit_group else source['name'] + '/' + group),
                                    width=w, height=h,
                                    sha256=digest_file(path), rois=rois))
            except (OSError, ValueError) as exc:
                rejected.append(dict(path=str(path), reason=str(exc)))

    # A duplicate linking two videos must not put their remaining frames on
    # opposite sides of the split. Union before dropping duplicate records.
    parents = {}
    def find(group):
        parents.setdefault(group, group)
        if parents[group] != group:
            parents[group] = find(parents[group])
        return parents[group]
    by_hash = {}
    unique = []
    for item in entries:
        group = find(item['group'])
        previous = by_hash.get(item['sha256'])
        if previous is None:
            by_hash[item['sha256']] = item
            unique.append(item)
        else:
            a, b = sorted((find(previous['group']), group))
            parents[b] = a
            if item['rois'] and not previous['rois']:
                previous['rois'] = item['rois']
    groups = sorted({find(item['group']) for item in unique})
    if len(groups) < 2:
        raise ValueError('need at least two independent groups for train/validation isolation')
    fraction = float(cfg.get('val_fraction', .05))
    if not 0 < fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    # 2026-10-19:允许按源覆盖验证集比例。连续帧源(如 uhd)用全局 5% 会切走太多,
    # 而它又必须出现在验证集里(否则验证分布缺一大块);给该源一个更小的
    # val_fraction 即可,例如 uhd 用 0.01。
    source_fraction = {s['name']: float(s.get('val_fraction', fraction))
                       for s in cfg['sources']}
    for name, value in source_fraction.items():
        if not 0 < value < 1:
            raise ValueError(f'{name}.val_fraction must be between zero and one')
    scores = {g: int(hashlib.sha256((str(cfg['seed']) + '/' + g).encode()).hexdigest()[:16], 16) / 2**64
              for g in groups}
    groups_by_source = {}
    for item in unique:
        # 必须走 find() 归到并查集的根名字:去重时同一内容的多个组名会被合并,
        # scores 是按根名字建的。直接存 item['group'] 会 KeyError。
        groups_by_source.setdefault(item['source'], set()).add(find(item['group']))
    val_groups = set()
    for source, owned in groups_by_source.items():
        # 只有一个组的源无法切分:无论它的哈希分数落在哪,都必须整组进 train。
        # (gameplay / frames_2k 用 group_by=directory,整个源就是一组;没有这条守卫
        #  会让整个源被判进 val,训练里一条都不剩 —— 2026-10-19 实际踩到过。)
        if len(owned) < 2:
            continue
        share = source_fraction.get(source, fraction)
        chosen = {g for g in owned if scores[g] < share}
        if not chosen:
            chosen.add(min(owned, key=scores.get))
        if len(chosen) == len(owned) and len(owned) > 1:
            chosen.remove(max(owned, key=scores.get))
        val_groups |= chosen      # 组是跨源共享的:任一源选中即整组进验证,避免泄漏
    for item in unique:
        item['split_group'] = find(item['group'])
        item['split'] = 'val' if item['split_group'] in val_groups else 'train'
    source_counts = {}
    for source in cfg['sources']:
        items = [x for x in unique if x['source'] == source['name']]
        source_counts[source['name']] = dict(
            retained=len(items), groups=len({x['split_group'] for x in items}),
            train=sum(x['split'] == 'train' for x in items),
            val=sum(x['split'] == 'val' for x in items),
            roi_images=sum(bool(x['rois']) for x in items))
    return dict(version=1, seed=cfg['seed'], sources=cfg['sources'], entries=unique,
                source_counts=source_counts,
                rejected=rejected, counts=dict(scanned=len(entries), retained=len(unique),
                duplicates=len(entries) - len(unique), rejected=len(rejected),
                train=sum(x['split'] == 'train' for x in unique),
                val=sum(x['split'] == 'val' for x in unique),
                roi_images=sum(bool(x['rois']) for x in unique)),
                split_policy='source group, unioned by exact-byte duplicates; seed-hash holdout')


class SingleFrameSource:
    def __init__(self, manifest_path, cfg):
        self.cfg = cfg
        with open(manifest_path, encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('version') != 1:
            raise ValueError('unsupported single-frame manifest')
        self.entries = manifest['entries']
        self.train = [x for x in self.entries if x['split'] == 'train']
        self.val = sorted((x for x in self.entries if x['split'] == 'val'),
                          key=lambda x: hashlib.sha256(x['path'].encode()).hexdigest())
        if not self.train or not self.val:
            raise ValueError('manifest needs both train and validation entries')
        for key in ('split_group', 'sha256'):
            if {x[key] for x in self.train}.intersection(x[key] for x in self.val):
                raise ValueError('manifest leaks ' + key + ' across train/validation')
        excluded = set(cfg.get('exclude_components', ['proxy_proc']))
        if any(excluded.intersection(Path(x['path']).parts) for x in self.entries):
            raise ValueError('excluded source present in manifest')
        self.weights = {s['name']: float(s['weight']) for s in cfg['sources']}
        self.sizes = list(map(int, cfg['size']))
        if not self.sizes or min(self.sizes) < 33 or len(self.sizes) != len(cfg['size_weights']):
            raise ValueError('size/size_weights must have matching lengths and valid dimensions >=33')
        span = cfg.get('crop_scale', [1.0, 1.5])
        if len(span) != 2 or not 0 < span[0] <= span[1]:
            raise ValueError('crop_scale must be an ordered positive interval')
        if not cfg['context_buckets'] or any(len(b) != 2 or min(b) < 33 for b in cfg['context_buckets']):
            raise ValueError('context buckets must contain valid width/height pairs >=33')
        self.size_prob = self._prob(cfg['size_weights'])
        self.mode_prob = self._prob([cfg['sampling'][m] for m in MODES])
        self.pools = {}
        for size in self.sizes:
            for roi in (False, True):
                pool = {}
                for item in self.train:
                    if min(item['width'], item['height']) < size or (roi and not item['rois']):
                        continue
                    pool.setdefault(item['source'], {}).setdefault(item['split_group'], []).append(item)
                self.pools[size, roi] = pool
        self.all_pool = {}
        for item in self.train:
            self.all_pool.setdefault(item['source'], {}).setdefault(item['split_group'], []).append(item)
        if cfg['sampling'].get('roi_native', 0) and not any(self.pools[s, True] for s in self.sizes):
            raise ValueError('ROI sampling requested but no eligible training ROI; prepare with --detect-faces or --roi-jsonl')
        for size in self.sizes:
            if not self.pools[size, False]:
                raise ValueError('no image large enough for native crop size ' + str(size))

    @staticmethod
    def _prob(values):
        p = np.asarray(values, dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError('sampling weights must be finite, nonnegative and nonzero')
        return p / p.sum()

    def _choose(self, pool, rng):
        names = sorted(n for n in pool if self.weights.get(n, 0) > 0)
        if not names:
            raise ValueError('no eligible positively weighted source')
        name = names[int(rng.choice(len(names), p=self._prob([self.weights[n] for n in names])))]
        groups = sorted(pool[name])
        group = pool[name][groups[int(rng.integers(len(groups)))]]
        return group[int(rng.integers(len(group)))]

    def sample(self, index):
        rng = np.random.default_rng([int(self.cfg['seed']), int(index)])
        size = self.sizes[int(rng.choice(len(self.sizes), p=self.size_prob))]
        mode = MODES[int(rng.choice(len(MODES), p=self.mode_prob))]
        if mode in ('roi_native', 'random_native'):
            pool = self.pools[size, mode == 'roi_native']
            if not pool and mode == 'roi_native':
                # Keep the ROI category honest: choose a smaller eligible crop,
                # never silently report a random background crop as an ROI.
                size = next(s for s in sorted(self.sizes) if self.pools[s, True])
                pool = self.pools[size, True]
        else:
            pool = self.all_pool
        item = self._choose(pool, rng)
        image = read_rgb(item['path'])
        original = image.size
        resized_size = original
        scale, box, roi = 1.0, None, None
        if mode == 'context':
            image, scale = self._context(image)
            resized_size = image.size
        else:
            if mode == 'scaled_crop':
                span = float(rng.uniform(*self.cfg.get('crop_scale', [1.0, 1.5])))
                scale = max(1.0 / span, size / min(image.size))
                image = image.resize((max(size, round(image.width * scale)),
                                      max(size, round(image.height * scale))), RESAMPLE)
                resized_size = image.size
            w, h = image.size
            if mode == 'roi_native':
                roi = item['rois'][int(rng.integers(len(item['rois']))) ]
                cx = (roi[0] + roi[2]) / 2 + rng.uniform(-.15, .15) * size
                cy = (roi[1] + roi[3]) / 2 + rng.uniform(-.15, .15) * size
                x = int(np.clip(round(cx - size / 2), 0, w - size))
                y = int(np.clip(round(cy - size / 2), 0, h - size))
            else:
                x, y = int(rng.integers(w - size + 1)), int(rng.integers(h - size + 1))
            box = [x, y, x + size, y + size]
            image = image.crop(box)  # Native modes NEVER pass through resize.
        return self._pack(item, image, mode, index, dict(original_size=list(original),
                          resized_size=list(resized_size), crop_box=box, scale=scale, roi=roi))

    def _context(self, image):
        # Fit inside a finite aspect bucket; return the actual valid rectangle.
        # The teacher's own mirrored geometry padding handles the remainder.
        # There is no artificial letterbox content to supervise.
        aspect = image.width / image.height
        bucket = min(self.cfg['context_buckets'], key=lambda b: abs(math.log(aspect / (b[0] / b[1]))))
        scale = min(bucket[0] / image.width, bucket[1] / image.height, 1.0)
        scale = max(scale, 33 / min(image.size))
        return image.resize((max(33, round(image.width * scale)),
                             max(33, round(image.height * scale))), RESAMPLE), scale

    def validation(self, index, mode='native'):
        item = self.val[index]
        image = read_rgb(item['path'])
        original = image.size
        box = None
        if mode == 'crop':
            sizes = [s for s in self.sizes if s <= min(image.size)]
            if sizes:
                side = max(sizes)
                x, y = (image.width - side) // 2, (image.height - side) // 2
                box = [x, y, x + side, y + side]
                image = image.crop(box)
        elif mode != 'native':
            raise ValueError('validation mode must be crop or native')
        return self._pack(item, image, 'val_' + mode, 1000000000 + index,
                          dict(original_size=list(original), crop_box=box, scale=1.0, roi=None))

    def _pack(self, item, image, mode, index, transform):
        proxy = np.asarray(image, dtype=np.float32) / 255.0
        h, w = proxy.shape[:2]
        geometry = geometry_from_valid(w, h)
        rng = np.random.default_rng([int(self.cfg['seed']), int(index), 91])
        # Map original boxes to the exact realized pixels (resize rounding can
        # give slightly different horizontal/vertical ratios).
        box = transform['crop_box']
        ow, oh = transform['original_size']
        rw, rh = transform.get('resized_size', (ow, oh))
        sx, sy = rw / ow, rh / oh
        left, top = (box[:2] if box else (0, 0))
        valid_rois = []
        for x0, y0, x1, y1 in item['rois']:
            mapped = [max(0, int(math.floor(x0 * sx - left))), max(0, int(math.floor(y0 * sy - top))),
                      min(w, int(math.ceil(x1 * sx - left))), min(h, int(math.ceil(y1 * sy - top)))]
            if mapped[0] < mapped[2] and mapped[1] < mapped[3]:
                valid_rois.append(mapped)
        return dict(proxy=proxy, geometry=geometry, hw=(h, w),
                    noise_seed=int(rng.integers(1 << 30)),
                    controls=dict(self.cfg['controls']),
                    metadata=dict(path=item['path'], source=item['source'],
                                  split_group=item['split_group'], mode=mode,
                                  sample_index=index, valid_hw=[h, w], valid_rois=valid_rois, **transform))
