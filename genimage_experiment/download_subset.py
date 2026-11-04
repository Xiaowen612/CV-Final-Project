"""Read official GenImage split ZIPs using HTTP ranges, without full archives.

No credential is required for these publicly shared official download files.
Every extracted member is checked against its ZIP CRC32 and byte length.
"""
import argparse
import concurrent.futures
import hashlib
import json
import random
import struct
import threading
import time
import zlib
from pathlib import Path, PurePosixPath

import requests
from bs4 import BeautifulSoup

FOLDERS = {'sd14': '12xighYOtu-ryfYEUnNrSeZqrxT8P08Zy', 'biggan': '1ajlTuN34gLyJWxRQ6NyUcnkfrS8QEVKt',
           'adm': '1-9o163XaC-7L8Ch9-r7dLxbY_yjN8SVj', 'glide': '1H2_4VPlla4OuKYU2CbMYx-8X0nbGJ2Mc'}
NAMES = {'sd14': 'Stable Diffusion V1.4', 'biggan': 'BigGAN', 'adm': 'ADM', 'glide': 'GLIDE'}
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def get_range(file_id, start=None, length=None, suffix=None):
    header = f'bytes=-{suffix}' if suffix else f'bytes={start}-{start+length-1}'
    url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
    for attempt in range(5):
        try:
            with session().get(url, headers={'Range': header}, stream=True, timeout=(20, 90)) as response:
                if response.status_code != 206:
                    raise IOError(f'Expected HTTP 206, got {response.status_code}; refusing full archive download.')
                content_range = response.headers['Content-Range']
                total = int(content_range.split('/')[-1])
                got_start, got_end = map(int, content_range.split(' ')[1].split('/')[0].split('-'))
                expected_start = max(0, total-suffix) if suffix else start
                expected_length = total-expected_start if suffix else min(length, total-start)
                if got_start != expected_start or got_end-got_start+1 != expected_length:
                    raise IOError('Server returned the wrong byte range.')
                data = response.content
                if len(data) != expected_length:
                    raise IOError('Truncated range response.')
                return data, total
        except (requests.RequestException, IOError, KeyError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


class RemoteSplitZip:
    def __init__(self, generator, cache):
        self.generator = generator
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        index_file = self.cache / f'{generator}_volumes.json'
        if index_file.exists():
            self.volumes = json.loads(index_file.read_text())
        else:
            response = session().get('https://drive.google.com/drive/folders/'+FOLDERS[generator], timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            found = {}
            for element in soup.select('[data-id]'):
                name = element.get_text(' ', strip=True)
                if name.endswith('.zip') or ('.z' in name and name.rsplit('.z', 1)[-1].isdigit()):
                    found[name] = element['data-id']
            self.volumes = [{'name': n, 'id': found[n]} for n in sorted(found)]
            if not self.volumes or not self.volumes[-1]['name'].endswith('.zip'):
                raise IOError('Could not discover official split ZIP volumes.')
            index_file.write_text(json.dumps(self.volumes, indent=2))
        self.sizes = {}

    def read(self, disk, offset, length):
        chunks = []
        while length:
            if disk not in self.sizes:
                _, self.sizes[disk] = get_range(self.volumes[disk]['id'], start=0, length=4)
            n = min(length, self.sizes[disk] - offset)
            if n <= 0:
                raise IOError('Invalid ZIP volume offset.')
            content, _ = get_range(self.volumes[disk]['id'], start=offset, length=n)
            chunks.append(content)
            length -= n
            disk, offset = disk+1, 0
        return b''.join(chunks)

    def entries(self):
        index = self.cache / f'{self.generator}_members.json'
        if index.exists():
            return json.loads(index.read_text())
        tail, total = get_range(self.volumes[-1]['id'], suffix=65536)
        end = tail.rfind(b'PK\x05\x06')
        if end < 0:
            raise IOError('ZIP end-of-directory missing.')
        eocd = struct.unpack_from('<4s4H2IH', tail, end)
        disk, cd_size, cd_offset = eocd[2], eocd[5], eocd[6]
        locator = tail.rfind(b'PK\x06\x07', 0, end)
        if locator >= 0:
            _, record_disk, record_offset, _ = struct.unpack_from('<4sIQI', tail, locator)
            record = self.read(record_disk, record_offset, 56)
            values = struct.unpack('<4sQ2H2I4Q', record)
            disk, cd_size, cd_offset = values[5], values[8], values[9]
        print(f'{self.generator}: reading {cd_size/2**20:.1f} MiB central directory', flush=True)
        central = self.read(disk, cd_offset, cd_size)
        position, entries = 0, []
        while position < len(central) and central[position:position+4] == b'PK\x01\x02':
            v = struct.unpack_from('<4s6H3I5H2I', central, position)
            name_len, extra_len, comment_len = v[10:13]
            name = central[position+46:position+46+name_len].decode('utf-8' if v[3] & 2048 else 'cp437')
            entry = dict(name=name, flags=v[3], compression=v[4], crc=v[7], compressed=v[8], size=v[9], disk=v[13], offset=v[16])
            extra = central[position+46+name_len:position+46+name_len+extra_len]
            ep = 0
            while ep+4 <= len(extra):
                kind, length = struct.unpack_from('<HH', extra, ep)
                data = extra[ep+4:ep+4+length]
                if kind == 1:
                    cursor = 0
                    for key in ['size', 'compressed', 'offset', 'disk']:
                        sentinel = 65535 if key == 'disk' else 4294967295
                        if entry[key] == sentinel:
                            fmt = '<I' if key == 'disk' else '<Q'
                            entry[key] = struct.unpack_from(fmt, data, cursor)[0]
                            cursor += struct.calcsize(fmt)
                ep += 4+length
            if Path(name).suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                entries.append(entry)
            position += 46+name_len+extra_len+comment_len
        if not entries:
            raise IOError('No images found in central directory.')
        index.write_text(json.dumps(entries, separators=(',', ':')))
        return entries

    def extract(self, entry, output):
        output = Path(output)
        if output.exists():
            raw = output.read_bytes()
            if len(raw) == entry['size'] and zlib.crc32(raw) == entry['crc']:
                return
        # Local header is small; range includes its actual filename and extra field.
        header = self.read(entry['disk'], entry['offset'], 30)
        values = struct.unpack('<4s5H3I2H', header)
        if values[0] != b'PK\x03\x04' or entry['flags'] & 1:
            raise IOError('Invalid or encrypted ZIP member.')
        prefix = 30+values[-2]+values[-1]
        # Read from local-header start so cross-volume boundaries are handled once.
        raw = self.read(entry['disk'], entry['offset'], prefix+entry['compressed'])[prefix:]
        if entry['compression'] == 8:
            raw = zlib.decompress(raw, -15)
        elif entry['compression'] != 0:
            raise IOError(f"Unsupported compression {entry['compression']}")
        if len(raw) != entry['size'] or zlib.crc32(raw) != entry['crc']:
            raise IOError('Extracted member failed ZIP integrity checks.')
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix+'.partial')
        temporary.write_bytes(raw)
        temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/GenImage')
    parser.add_argument('--cache', default='download_cache')
    parser.add_argument('--train-pool', type=int, default=220)
    parser.add_argument('--test-pool', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    selections = []
    for generator in FOLDERS:
        archive = RemoteSplitZip(generator, args.cache)
        entries = archive.entries()
        rng = random.Random(f'{args.seed}/{generator}')
        for split in (['train', 'val'] if generator in {'sd14', 'biggan'} else ['val']):
            for category in ['ai', 'nature']:
                candidates = [e for e in entries if f'/{split}/{category}/' in e['name']]
                rng.shuffle(candidates)
                n = args.train_pool if split == 'train' else args.test_pool
                if len(candidates) < n:
                    raise ValueError(f'{generator}/{split}/{category}: only {len(candidates)} candidates')
                jobs = []
                for entry in candidates[:n]:
                    relative = entry['name'].split(f'/{split}/{category}/', 1)[1]
                    if '..' in PurePosixPath(relative).parts or PurePosixPath(relative).is_absolute():
                        raise ValueError('Unsafe archive path.')
                    output = Path(args.root) / NAMES[generator] / split / category / relative
                    selections.append(dict(generator=generator, official_member=entry['name'], crc32=entry['crc'],
                                           local_path=str(output), official_folder_id=FOLDERS[generator]))
                    jobs.append((entry, output))
                with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
                    futures = [pool.submit(archive.extract, e, p) for e, p in jobs]
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        future.result()
                        if (i+1) % 50 == 0 or i+1 == len(futures):
                            print(f'{generator}/{split}/{category}: {i+1}/{len(futures)}', flush=True)
    Path(args.root, 'download_provenance.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
