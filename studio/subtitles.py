"""Strict SRT parsing and timestamp-based audio/text pairing."""
import html
import re


def timestamp(value):
    match = re.fullmatch(r'(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{3})', value.strip())
    if not match:
        raise ValueError(f'Invalid SRT timestamp: {value}')
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse(text, duration):
    rows = []
    last_end = 0
    for block in re.split(r'\n\s*\n', text.lstrip('\ufeff').replace('\r\n', '\n').strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if lines[0].strip().isdigit():
            lines = lines[1:]
        if not lines or '-->' not in lines[0]:
            raise ValueError('Invalid SRT block: missing timestamp range.')
        start, end = map(timestamp, lines[0].split('-->'))
        caption = html.unescape(re.sub(r'<[^>]+>', '', ' '.join(lines[1:]))).strip()
        if not caption or start < last_end - .0001 or end <= start or end > duration + .05:
            raise ValueError('SRT must contain text, ordered non-overlapping timestamps, and fit the selected audio duration.')
        if end - start > 12:
            raise ValueError('A subtitle cue exceeds 12 seconds. Split that cue before importing.')
        rows.append({'start': start, 'end': min(end, duration), 'text': caption})
        last_end = end
    if not rows:
        raise ValueError('The subtitle file is empty.')
    return rows


def grouped(rows):
    result = []
    for row in rows:
        if (result and result[-1]['end'] - result[-1]['start'] < 6
                and row['end'] - result[-1]['start'] <= 12
                and row['start'] - result[-1]['end'] <= 1.25):
            result[-1]['end'] = row['end']
            result[-1]['text'] += ' ' + row['text']
        else:
            result.append(dict(row))
    return result


def word_error(reference, hypothesis):
    def words(value):
        return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", value.lower().replace('’', "'"))
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / max(1, len(ref))
