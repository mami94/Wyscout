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
from sklearn.preprocessing import RobustScaler, MinMaxScaler

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

# Farkli liglerin genel seviyesini tek bir katsayiyla T skoruna yansitmak icin.
# 1.0 = referans seviye. Dataset id'sine (convert_dataset.py'a verdigin 2. parametre,
# orn. "pol1") gore eslesir. Burada olmayan bir lig icin katsayi otomatik 1.0 kabul edilir.
# Degerler tahmini/taslak - birlikte netlestirene kadar 1.0 (etkisiz) birakildi.
LEAGUE_STRENGTH = {
    "pol1": 1.0,
    "por2": 1.0,
    "italy2": 1.0,
    "greece1": 1.0,
    "scotland1": 1.0,
    "tunisia1": 1.0,
    "morocco1": 1.0,
    "egypt1": 1.0,
    "france3": 1.0,
    "germany3": 1.0,
    "cyprus1": 1.0,
    "hungary1": 1.0,
    "georgia1": 1.0,
    "bulgaria1": 1.0,
    "korea1": 1.0,
    "slovenia1": 1.0,
    "slovakia1": 1.0,
    "chile1": 1.0,
    "ecuador1": 1.0,
    "uruguay1": 1.0,
    "romania1": 1.0,
}


def _foot_missing(f):
    if pd.isna(f): return True
    s = str(f).strip().lower()
    return s in ('', 'unknown', 'nan', '0')


def _height_compat(h1, h2):
    h1 = 0 if pd.isna(h1) else h1
    h2 = 0 if pd.isna(h2) else h2
    if h1 > 0 and h2 > 0 and h1 != h2:
        return False
    return True


def _age_compat(a1, a2):
    if pd.isna(a1) or pd.isna(a2):
        return True
    return abs(a1 - a2) <= 1


def _foot_compat(f1, f2):
    if _foot_missing(f1) or _foot_missing(f2):
        return True
    return str(f1).strip().lower() == str(f2).strip().lower()


def merge_duplicate_players(raw):
    """
    Ayni isimli birden fazla satiri, GERCEKTEN ayni oyuncu oldugunu dusundugumuzde
    (boy celismiyor, yas farki <=1, ayak celismiyor) TEK satirda birlestirir.
    Isim ayni ama boy/yas/ayak celisiyorsa (orn. "Bruno Silva" gibi yaygin isim
    tasiyan farkli oyuncular) AYRI birakir - yanlislikla farkli kisileri
    birlestirmemek icin.
    """
    per90_cols = [c for c in raw.columns if 'per 90' in c]
    pct_cols = [c for c in raw.columns if c.strip().endswith('%')]
    avg_cols = [c for c in raw.columns if c.strip().lower().startswith('average') and c not in per90_cols]
    weighted_cols = list(set(per90_cols + pct_cols + avg_cols))

    sum_cols = ['Matches played', 'Minutes played', 'Goals', 'xG', 'Assists', 'xA',
                'Yellow cards', 'Red cards', 'Non-penalty goals', 'Head goals', 'Shots',
                'Conceded goals', 'Shots against', 'Clean sheets', 'xG against',
                'Prevented goals', 'Penalties taken', 'PAdj Sliding tackles', 'PAdj Interceptions']
    sum_cols = [c for c in sum_cols if c in raw.columns]

    bio_cols = ['Age', 'Market value', 'Contract expires', 'Birth country', 'Passport country',
                'Foot', 'Height', 'Weight', 'On loan', 'Position', 'Team']
    bio_cols = [c for c in bio_cols if c in raw.columns]

    merged_rows = []
    merge_log = []
    skip_log = []

    for name, group in raw.groupby('Player', sort=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0])
            continue

        idxs = list(group.index)
        parent = {i: i for i in idxs}
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb: parent[ra] = rb

        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                a, b = raw.loc[idxs[i]], raw.loc[idxs[j]]
                if (_height_compat(a.get('Height'), b.get('Height')) and
                    _age_compat(a.get('Age'), b.get('Age')) and
                    _foot_compat(a.get('Foot'), b.get('Foot'))):
                    union(idxs[i], idxs[j])

        clusters = {}
        for i in idxs:
            clusters.setdefault(find(i), []).append(i)

        for cluster_idxs in clusters.values():
            if len(cluster_idxs) == 1:
                merged_rows.append(raw.loc[cluster_idxs[0]])
                continue

            rows = raw.loc[cluster_idxs].copy()
            rows = rows.sort_values('Minutes played', ascending=False)
            primary = rows.iloc[0].copy()
            total_minutes = rows['Minutes played'].fillna(0).sum()

            for c in sum_cols:
                primary[c] = rows[c].fillna(0).sum()
            for c in weighted_cols:
                if c in rows.columns and total_minutes > 0:
                    w = rows['Minutes played'].fillna(0)
                    vals = rows[c].fillna(0)
                    primary[c] = (vals * w).sum() / total_minutes
            for c in bio_cols:
                if pd.isna(primary.get(c)) or primary.get(c) in (0, '', 'unknown'):
                    for _, r in rows.iterrows():
                        if not pd.isna(r.get(c)) and r.get(c) not in (0, '', 'unknown'):
                            primary[c] = r.get(c)
                            break

            teams_in_window = [str(t) for t in rows['Team within selected timeframe'].dropna().unique() if str(t).strip()]
            if len(teams_in_window) > 1:
                primary['Team within selected timeframe'] = ' / '.join(teams_in_window)
            elif teams_in_window:
                primary['Team within selected timeframe'] = teams_in_window[0]

            merged_rows.append(primary)
            merge_log.append(f"  - {name}: {len(cluster_idxs)} kayit birlestirildi ({total_minutes:.0f} dk toplam)")

        if len(clusters) > 1:
            skip_log.append(f"  - {name}: {len(clusters)} farkli oyuncu olarak ayri birakildi (yas/boy/ayak celisiyor)")

    if merge_log:
        print(f"[merge] {len(merge_log)} oyuncu birlestirildi:")
        for line in merge_log: print(line)
    if skip_log:
        print(f"[merge] {len(skip_log)} isim celismesi ayri birakildi (farkli kisiler):")
        for line in skip_log: print(line)

    return pd.DataFrame(merged_rows).reset_index(drop=True)


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
    raw = merge_duplicate_players(raw)
    team_fallback = raw.set_index('Player')['Team within selected timeframe'].to_dict()
    parent_club = raw.set_index('Player')['Team'].to_dict()

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
    # alt %20 0.50 carpanla degerlendirilir (once 0.35'ti, cok sertti).
    def playtime_multiplier(minutes_series):
        q20, q40, q60, q80 = minutes_series.quantile([0.2, 0.4, 0.6, 0.8])
        def mult(m):
            if m >= q80: return 1.00
            if m >= q60: return 0.95
            if m >= q40: return 0.85
            if m >= q20: return 0.70
            return 0.50
        return minutes_series.apply(mult)

    data['PlayTimeMult'] = data.groupby('Pos.group')['Minutes played'].transform(playtime_multiplier)

    # RobustScaler: medyan ve IQR kullanir, StandardScaler'in aksine
    # xG / crosses / smart passes gibi sagdan carpik, aykiri-degerli
    # metriklerde tek bir ekstrem performans skoru bozmaz.
    scaler = RobustScaler()
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

    # Alt kategori skorlari (DQ, AA, BC...) artik TUM oyuncu havuzuna gore degil,
    # her pozisyon grubunun KENDI icinde 0-100'e normalize ediliyor. Boylece bir
    # CB'nin Crossing skoru wingerlarla degil, diger CB'lerle kiyaslaniyor.
    def norm_group_minmax(s):
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min()) * 100
        return pd.Series(50.0, index=s.index)

    for score in category_scores.keys():
        data_with_scores[score] = data_with_scores.groupby('Pos.group')[score].transform(norm_group_minmax)

    # ---- Final skor: agirlikli ARITMETIK ortalama yerine agirlikli GEOMETRIK
    # ortalama kullaniyoruz. Aritmetik ortalamada tek bir cok guclu kategori,
    # baska bir kategorideki zayifligi rahatca kapatabiliyordu (orn. 90 + 10 ortalamasi
    # hala 50). Geometrik ortalamada ayni oyuncu cok daha asagida kalir (sqrt(90*10)=30),
    # yani "her yonden dengeli olmayan" oyuncular T'de daha fazla cezalandirilir.
    # score+1 kullanimi log(0) patlamasin diye (0-100 skalada etkisi ihmal edilebilir).
    def calc_T_raw(row):
        pos = row['Pos.group']
        pw = POSITION_CATEGORY_WEIGHTS.get(pos, {c: 1 for c in WEIGHTS})
        tw = sum(pw.values()) or 1
        log_sum = sum(pw.get(c, 0) * np.log(max(row.get(f'{c} Score', 0), 0) + 1) for c in WEIGHTS.keys())
        s = np.exp(log_sum / tw) - 1
        return max(s, 0) * row['PlayTimeMult']

    data_with_scores['T_raw'] = data_with_scores.apply(calc_T_raw, axis=1)

    # ---- Toplam skoru (T) her pozisyon grubu icinde ayri ayri 0-100'e normalize et.
    # Boylece her ligde her mevkinin en iyisi 100, en kotusu 0'a yakin olur ve
    # farkli mevkiler/ligler arasi T karsilastirmasi adil olur.
    def norm_group(s):
        if s.max() > s.min():
            return (s - s.min()) / (s.max() - s.min()) * 100
        return pd.Series(50.0, index=s.index)

    data_with_scores['T'] = data_with_scores.groupby('Pos.group')['T_raw'].transform(norm_group)

    # ---- League Strength: farkli liglerin genel seviyesini T'ye uygulanan tek bir
    # katsayi ile ayarlamak icin ayrilmis alan. LEAGUE_STRENGTH sozlugunde bu
    # dataset'in "key"ine (orn. "pol1") karsilik gelen katsayi varsa T bununla
    # carpilir (0-100 pozisyon ici normalizasyondan SONRA uygulanir, boylece bir
    # ligin en iyisi hala kendi ligi icinde 100'du ama farkli ligler karsilastirildiginda
    # gercek seviyeleri yansitir). Katsayi belirtilmemisse 1.0 (degisiklik yok) kullanilir.
    league_strength = LEAGUE_STRENGTH.get(key, 1.0)
    data_with_scores['T'] = (data_with_scores['T'] * league_strength).round(2)

    # ---- Value Score: T skorunu piyasa degerine (milyon euro basina) bolerek
    # "parasina gore performans" olcusu veriyor. Piyasa degeri 0/bilinmiyorsa None.
    def calc_value_score(row):
        mv = row['Market value']
        if not mv or mv <= 0:
            return None
        return round(row['T'] / (mv / 1_000_000), 2)

    data_with_scores['ValueScore'] = data_with_scores.apply(calc_value_score, axis=1)

    out = data_with_scores[['Player', 'Team', 'Pos.group', 'Age', 'Contract expires', 'Market value',
                             'Minutes played', 'Foot', 'Height'] + list(category_scores.keys()) + ['T', 'ValueScore']].copy()
    out.rename(columns=RENAME, inplace=True)
    out['T'] = out['T'].round(1)
    out['ValueScore'] = out['ValueScore'].round(2)
    out[list(RENAME.values())] = out[list(RENAME.values())].round(1)
    out = out.sort_values(by='T', ascending=False)

    def is_blank(v):
        return v in (0, '0', '', None) or (isinstance(v, float) and pd.isna(v))

    def resolve_team(row):
        # Asil gosterilecek takim: bu ligde GERCEKTE oynadigi kulup
        # ("Team within selected timeframe"). "Team" alani cogu zaman
        # kiralik oyuncularda sahip/ana kulubu (bazen yurt disindan) gosteriyor.
        within = team_fallback.get(row['Player'])
        parent = parent_club.get(row['Player'])
        primary = within if isinstance(within, str) and within.strip() and not is_blank(within) else (
            parent if isinstance(parent, str) and not is_blank(parent) else '—')
        on_loan_from = None
        if (isinstance(parent, str) and not is_blank(parent) and isinstance(within, str)
                and not is_blank(within) and parent.strip() != within.strip()
                and '/' not in within):  # birlesmis (transfer) satirlarda loan etiketi gosterme
            on_loan_from = parent.strip()
        return pd.Series([primary, on_loan_from])

    out[['Team', 'OnLoanFrom']] = out.apply(resolve_team, axis=1)

    out['Contract expires'] = out['Contract expires'].apply(lambda v: '' if (v == 0 or pd.isna(v)) else str(v)[:10])
    out['Market value'] = out['Market value'].fillna(0).astype(int)
    out['Minutes played'] = out['Minutes played'].astype(int)
    out['Age'] = out['Age'].astype(int)
    out['Height'] = out['Height'].fillna(0).astype(int)

    records = out.to_dict(orient='records')
    for rec in records:
        if isinstance(rec.get('ValueScore'), float) and np.isnan(rec['ValueScore']):
            rec['ValueScore'] = None
        if rec.get('OnLoanFrom') is None or (isinstance(rec.get('OnLoanFrom'), float) and np.isnan(rec['OnLoanFrom'])):
            rec['OnLoanFrom'] = None

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
