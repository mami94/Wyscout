"""
convert_dataset.py
-------------------
Bir Wyscout export dosyasini (Pol1.xlsx formatinda) SCOUT SITESI icin
gzip'li JSON'a cevirir.

Kullanim:
    python convert_dataset.py Pol1.xlsx pol1 "Polonya 1. Lig - 2025/26"
    python convert_dataset.py Tur1.xlsx tur1 "Turkiye 1. Lig - 2025/26"

Bu dosyayi index.html ile AYNI klasorde tut. Uretilen dosyalar, bu dosyanin
yaninda yer alan data/ klasorune yazilir ve data/manifest.json otomatik
guncellenir (ayni id varsa uzerine yazar, yoksa listeye ekler).
"""
import sys, json, gzip, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Bu dosya site klasorunun icinde durur; veri dosyalari yaninda yer alan data/ klasorune yazilir.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

WEIGHTS = {
    "Defensive Score": {"Successful defensive actions per 90": 3, "PAdj Sliding tackles": 2,
                         "PAdj Interceptions": 2, "Succ Defensive Duels per 90": 3, "Shots blocked per 90": 1},
    "Aerial Ability": {"Aerial duels won, %": 3, "Aerial duels per 90": 2, "Succ Aerial Duels per 90": 2},
    "Ball Carrying": {"Accelerations per 90": 1, "Progressive runs per 90": 2, "Succ Dribbles per 90": 3,
                       "Succ Offensive Duels per 90": 2, "Touches in box per 90": 2},
    "Playmaking": {"Succ progressive passes per 90": 2, "Accurate forward passes, %": 2,
                    "Passes to final third per 90": 2, "Long passes per 90": 1, "Key passes per 90": 2,
                    "Smart passes per 90": 1, "Deep completions per 90": 2},
    "Advanced Playmaking": {"Succ Through passes per 90": 1, "Succ Passes to penalty area per 90": 2,
                             "Deep completions per 90": 2, "xA per 90": 2},
    "Shot Involvement": {"xG per 90": 3, "Non-penalty goals per 90": 2, "Shots on target, %": 2,
                          "Shots per 90": 1, "Touches in box per 90": 1},
    "Chance Creation": {"Shot assists per 90": 1, "xA per 90": 3, "Second assists per 90": 3,
                         "Third assists per 90": 3, "Key passes per 90": 2},
    "Crossing": {"Succ Crosses per 90": 2, "Deep completed crosses per 90": 2, "Crosses to goalie box per 90": 1},
    "Duels": {"Duels won, %": 2, "Duels per 90": 1},
}

POSITION_CATEGORY_WEIGHTS = {
    "CB": {"Defensive Score": 3, "Aerial Ability": 3, "Ball Carrying": 1, "Playmaking": 1,
           "Advanced Playmaking": 0, "Shot Involvement": 0, "Chance Creation": 0, "Crossing": 0, "Duels": 3},
    "FB": {"Defensive Score": 2, "Aerial Ability": 1, "Ball Carrying": 2, "Playmaking": 1,
           "Advanced Playmaking": 1, "Shot Involvement": 0, "Chance Creation": 1, "Crossing": 3, "Duels": 2},
    "DM": {"Defensive Score": 3, "Aerial Ability": 2, "Ball Carrying": 1, "Playmaking": 2,
           "Advanced Playmaking": 1, "Shot Involvement": 0, "Chance Creation": 1, "Crossing": 0, "Duels": 3},
    "CM": {"Defensive Score": 1, "Aerial Ability": 1, "Ball Carrying": 2, "Playmaking": 3,
           "Advanced Playmaking": 2, "Shot Involvement": 1, "Chance Creation": 2, "Crossing": 0, "Duels": 2},
    "AM": {"Defensive Score": 0, "Aerial Ability": 1, "Ball Carrying": 2, "Playmaking": 2,
           "Advanced Playmaking": 2, "Shot Involvement": 2, "Chance Creation": 3, "Crossing": 0, "Duels": 1},
    "W":  {"Defensive Score": 0, "Aerial Ability": 1, "Ball Carrying": 3, "Playmaking": 1,
           "Advanced Playmaking": 1, "Shot Involvement": 2, "Chance Creation": 2, "Crossing": 3, "Duels": 2},
    "S":  {"Defensive Score": 0, "Aerial Ability": 2, "Ball Carrying": 2, "Playmaking": 0,
           "Advanced Playmaking": 1, "Shot Involvement": 3, "Chance Creation": 1, "Crossing": 0, "Duels": 2},
}

RENAME = {"Defensive Score Score": "DQ", "Aerial Ability Score": "AA", "Ball Carrying Score": "BC",
          "Playmaking Score": "PM", "Advanced Playmaking Score": "APM", "Shot Involvement Score": "SI",
          "Chance Creation Score": "CC", "Crossing Score": "CRS", "Duels Score": "DUE"}

PARAMS = ["DQ", "AA", "BC", "PM", "APM", "SI", "CC", "CRS", "DUE"]

# Otomatik pozisyon siniflandirmasi bazen yanlis olabiliyor (orn. Wyscout'ta
# kanat olarak gecen ama aslinda ofansif orta saha oynayan oyuncular gibi).
# Boyle durumlarda oyuncu ismini birebir buraya ekleyip dogru mevki grubunu
# zorla atayabilirsin. Bu, T skorunun ve grup ortalamalarinin dogru mevkiye
# gore hesaplanmasini saglar.
POSITION_OVERRIDES = {
    "L. Ambros": "CM",
}

# Belirli bir oyuncunun belirli skor sutunlarini elle duzeltmek icin.
# Ornek: SCORE_OVERRIDES = {"J. Kolan": {"PM": 37.3, "DQ": 53.2}}
SCORE_OVERRIDES = {
    "J. Kolan": {"PM": 37.3, "DQ": 53.2},
}


def classify_position(mp):
    if mp in ['GK']: return 'GK'
    elif mp in ['CB', 'LCB', 'RCB']: return 'CB'
    elif mp in ['LB', 'RB', 'LWB', 'RWB', 'FB']: return 'FB'
    elif mp in ['CMF', 'LCMF', 'RCMF']: return 'CM'
    elif mp in ['DMF', 'CDM', 'LDMF', 'RDMF']: return 'DM'
    elif mp in ['AMF', 'CAMF']: return 'AM'
    elif mp in ['LW', 'RW', 'LM', 'RM', 'RWF', 'LWF', 'LAMF', 'RAMF']: return 'W'
    elif mp in ['CF', 'ST']: return 'S'
    else: return 'Other'


def process(path, key, label, min_minutes=600):
    raw = pd.read_excel(path)
    team_fallback = raw.set_index('Player')['Team within selected timeframe'].to_dict()

    data = raw.fillna(0)
    data['Position'] = data['Position'].astype(str)
    data['Main Position'] = data['Position'].apply(lambda x: x.split(',')[0].strip())
    data.insert(data.columns.get_loc('Position') + 1, 'Pos.group', data['Main Position'].apply(classify_position))
    if POSITION_OVERRIDES:
        mask = data['Player'].isin(POSITION_OVERRIDES.keys())
        data.loc[mask, 'Pos.group'] = data.loc[mask, 'Player'].map(POSITION_OVERRIDES)
    data = data[data['Minutes played'] > min_minutes]
    data = data[data['Position'] != "GK"]

    data['Succ Offensive Duels per 90']        = data['Offensive duels per 90'] * data['Offensive duels won, %'] / 100
    data['Succ Defensive Duels per 90']        = data['Defensive duels per 90'] * data['Defensive duels won, %'] / 100
    data['Succ Aerial Duels per 90']           = data['Aerial duels per 90'] * data['Aerial duels won, %'] / 100
    data['Succ Dribbles per 90']               = data['Dribbles per 90'] * data['Successful dribbles, %'] / 100
    data['Succ Crosses per 90']                = data['Crosses per 90'] * data['Accurate crosses, %'] / 100
    data['Succ progressive passes per 90']     = data['Progressive passes per 90'] * data['Accurate progressive passes, %'] / 100
    data['Succ Through passes per 90']         = data['Through passes per 90'] * data['Accurate through passes, %'] / 100
    data['Succ Passes to penalty area per 90'] = data['Passes to penalty area per 90'] * data['Accurate passes to penalty area, %'] / 100

    # ---- Oyun suresi cezasi: her pozisyon grubu icinde 5'e bolup (quintile)
    # az oynayanlari kademeli olarak cezalandiriyoruz. Ust %20 tam puan alir,
    # alt %20 sadece 0.35 carpanla degerlendirilir.
    def playtime_multiplier(minutes_series):
        q20, q40, q60, q80 = minutes_series.quantile([0.2, 0.4, 0.6, 0.8])
        def mult(m):
            if m >= q80: return 1.00
            if m >= q60: return 0.90
            if m >= q40: return 0.75
            if m >= q20: return 0.55
            return 0.35
        return minutes_series.apply(mult)

    data['PlayTimeMult'] = data.groupby('Pos.group')['Minutes played'].transform(playtime_multiplier)

    scaler = StandardScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    exclude_cols = ['Age', 'Market value', 'Minutes played', 'PlayTimeMult', 'Height']
    numeric_columns = [c for c in numeric_columns if c not in exclude_cols]
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    category_scores = {}
    for category, metrics in WEIGHTS.items():
        valid = {m: w for m, w in metrics.items() if m in data.columns}
        tw = sum(valid.values())
        category_scores[category + " Score"] = data[list(valid.keys())].apply(
            lambda row: sum(row[m] * w for m, w in valid.items()) / tw, axis=1)

    data_with_scores = pd.concat([data, pd.DataFrame(category_scores)], axis=1)
    minmax = MinMaxScaler(feature_range=(0, 100))
    for score in category_scores.keys():
        data_with_scores[score] = minmax.fit_transform(data_with_scores[[score]])

    def calc_T_raw(row):
        pos = row['Pos.group']
        pw = POSITION_CATEGORY_WEIGHTS.get(pos, {c: 1 for c in WEIGHTS})
        tw = sum(pw.values()) or 1
        s = sum(row.get(f'{c} Score', 0) * pw.get(c, 0) for c in WEIGHTS.keys()) / tw
        return s * row['PlayTimeMult']

    data_with_scores['T_raw'] = data_with_scores.apply(calc_T_raw, axis=1)

    # ---- Toplam skoru (T) her pozisyon grubu icinde ayri ayri 0-100'e normalize et.
    # Boylece her ligde her mevkinin en iyisi 100, en kotusu 0'a yakin olur ve
    # farkli mevkiler/ligler arasi T karsilastirmasi adil olur.
    def norm_group(s):
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min()) * 100
        return pd.Series(50.0, index=s.index)

    data_with_scores['T'] = data_with_scores.groupby('Pos.group')['T_raw'].transform(norm_group)

    out = data_with_scores[['Player', 'Team', 'Pos.group', 'Age', 'Contract expires', 'Market value',
                             'Minutes played', 'Foot', 'Height'] + list(category_scores.keys()) + ['T']].copy()
    out.rename(columns=RENAME, inplace=True)
    out = out.round(1)
    out = out.sort_values(by='T', ascending=False)

    def fix_team(row):
        if row['Team'] in (0, '0', None) or (isinstance(row['Team'], float) and pd.isna(row['Team'])):
            fb = team_fallback.get(row['Player'])
            return fb if isinstance(fb, str) and fb.strip() else '—'
        return row['Team']
    out['Team'] = out.apply(fix_team, axis=1)

    out['Contract expires'] = out['Contract expires'].apply(lambda v: '' if (v == 0 or pd.isna(v)) else str(v)[:10])
    out['Market value'] = out['Market value'].fillna(0).astype(int)
    out['Minutes played'] = out['Minutes played'].astype(int)
    out['Age'] = out['Age'].astype(int)
    out['Height'] = out['Height'].fillna(0).astype(int)

    records = out.to_dict(orient='records')

    if SCORE_OVERRIDES:
        for rec in records:
            fixes = SCORE_OVERRIDES.get(rec['Player'])
            if fixes:
                rec.update(fixes)

    return {"id": key, "label": label, "params": PARAMS, "players": records}


def main():
    if len(sys.argv) < 4:
        print('Kullanim: python convert_dataset.py <dosya.xlsx> <id> "<Etiket>"')
        sys.exit(1)
    xlsx_path, key, label = sys.argv[1], sys.argv[2], sys.argv[3]

    os.makedirs(DATA_DIR, exist_ok=True)
    payload = process(xlsx_path, key, label)

    json_path = os.path.join(DATA_DIR, f"{key}.json")
    gz_path = os.path.join(DATA_DIR, f"{key}.json.gz")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    with open(json_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        f_out.writelines(f_in)
    os.remove(json_path)  # sadece gz'i tutuyoruz, boyutu kucultmek icin

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest = [m for m in manifest if m["id"] != key]
    manifest.append({"id": key, "label": label, "file": f"data/{key}.json.gz"})
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"OK -> {gz_path}  ({len(payload['players'])} oyuncu)")
    print(f"manifest.json guncellendi ({len(manifest)} veri seti)")


if __name__ == "__main__":
    main()
