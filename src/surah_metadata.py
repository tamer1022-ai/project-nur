# Project Nur — Surah Metadata
# Bismillah
#
# Complete classification of all 114 surahs by revelation period.
# Based on the traditional Egyptian standard chronological ordering
# cross-referenced with Nöldeke's academic ordering.
#
# Categories:
#   EARLY_MECCAN  — First ~4 years of prophethood (Surahs of warning, Tawhid, Akhirah)
#   MIDDLE_MECCAN — ~Years 5-9 (Stories of previous prophets, detailed arguments)
#   LATE_MECCAN   — ~Years 10-13 (Consolidation, preparation for Hijrah)
#   MEDINAN       — After Hijrah (Legislation, community building, jihad)

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class RevelationPeriod(Enum):
    EARLY_MECCAN = "early_meccan"
    MIDDLE_MECCAN = "middle_meccan"
    LATE_MECCAN = "late_meccan"
    MEDINAN = "medinan"


@dataclass
class SurahMetadata:
    number: int
    name_arabic: str
    name_english: str
    name_transliterated: str
    verse_count: int
    revelation_period: RevelationPeriod
    chronological_order: int  # Traditional Egyptian standard
    meccan_or_medinan: str    # "Meccan" or "Medinan" (binary classification)


# Complete surah metadata — 114 surahs
# Chronological order and period classification based on:
# - Traditional Egyptian standard (used in most printed Mushafs)
# - Cross-referenced with al-Suyuti's al-Itqan
# - Nöldeke's academic chronology for secondary validation
#
# Period boundaries are approximate — scholars differ on exact classifications
# of transitional surahs. We use the majority scholarly position.

SURAH_METADATA: Dict[int, SurahMetadata] = {
    1: SurahMetadata(1, "الفاتحة", "The Opening", "Al-Fatihah", 7, RevelationPeriod.EARLY_MECCAN, 5, "Meccan"),
    2: SurahMetadata(2, "البقرة", "The Cow", "Al-Baqarah", 286, RevelationPeriod.MEDINAN, 87, "Medinan"),
    3: SurahMetadata(3, "آل عمران", "Family of Imran", "Ali 'Imran", 200, RevelationPeriod.MEDINAN, 89, "Medinan"),
    4: SurahMetadata(4, "النساء", "The Women", "An-Nisa", 176, RevelationPeriod.MEDINAN, 92, "Medinan"),
    5: SurahMetadata(5, "المائدة", "The Table Spread", "Al-Ma'idah", 120, RevelationPeriod.MEDINAN, 112, "Medinan"),
    6: SurahMetadata(6, "الأنعام", "The Cattle", "Al-An'am", 165, RevelationPeriod.LATE_MECCAN, 55, "Meccan"),
    7: SurahMetadata(7, "الأعراف", "The Heights", "Al-A'raf", 206, RevelationPeriod.LATE_MECCAN, 39, "Meccan"),
    8: SurahMetadata(8, "الأنفال", "The Spoils of War", "Al-Anfal", 75, RevelationPeriod.MEDINAN, 88, "Medinan"),
    9: SurahMetadata(9, "التوبة", "The Repentance", "At-Tawbah", 129, RevelationPeriod.MEDINAN, 113, "Medinan"),
    10: SurahMetadata(10, "يونس", "Jonah", "Yunus", 109, RevelationPeriod.LATE_MECCAN, 51, "Meccan"),
    11: SurahMetadata(11, "هود", "Hud", "Hud", 123, RevelationPeriod.LATE_MECCAN, 52, "Meccan"),
    12: SurahMetadata(12, "يوسف", "Joseph", "Yusuf", 111, RevelationPeriod.LATE_MECCAN, 53, "Meccan"),
    13: SurahMetadata(13, "الرعد", "The Thunder", "Ar-Ra'd", 43, RevelationPeriod.LATE_MECCAN, 96, "Meccan"),
    14: SurahMetadata(14, "إبراهيم", "Abraham", "Ibrahim", 52, RevelationPeriod.LATE_MECCAN, 72, "Meccan"),
    15: SurahMetadata(15, "الحجر", "The Rocky Tract", "Al-Hijr", 99, RevelationPeriod.LATE_MECCAN, 54, "Meccan"),
    16: SurahMetadata(16, "النحل", "The Bee", "An-Nahl", 128, RevelationPeriod.LATE_MECCAN, 70, "Meccan"),
    17: SurahMetadata(17, "الإسراء", "The Night Journey", "Al-Isra", 111, RevelationPeriod.LATE_MECCAN, 50, "Meccan"),
    18: SurahMetadata(18, "الكهف", "The Cave", "Al-Kahf", 110, RevelationPeriod.MIDDLE_MECCAN, 69, "Meccan"),
    19: SurahMetadata(19, "مريم", "Mary", "Maryam", 98, RevelationPeriod.MIDDLE_MECCAN, 44, "Meccan"),
    20: SurahMetadata(20, "طه", "Ta-Ha", "Ta-Ha", 135, RevelationPeriod.MIDDLE_MECCAN, 45, "Meccan"),
    21: SurahMetadata(21, "الأنبياء", "The Prophets", "Al-Anbiya", 112, RevelationPeriod.MIDDLE_MECCAN, 73, "Meccan"),
    22: SurahMetadata(22, "الحج", "The Pilgrimage", "Al-Hajj", 78, RevelationPeriod.MEDINAN, 103, "Medinan"),
    23: SurahMetadata(23, "المؤمنون", "The Believers", "Al-Mu'minun", 118, RevelationPeriod.MIDDLE_MECCAN, 74, "Meccan"),
    24: SurahMetadata(24, "النور", "The Light", "An-Nur", 64, RevelationPeriod.MEDINAN, 102, "Medinan"),
    25: SurahMetadata(25, "الفرقان", "The Criterion", "Al-Furqan", 77, RevelationPeriod.MIDDLE_MECCAN, 42, "Meccan"),
    26: SurahMetadata(26, "الشعراء", "The Poets", "Ash-Shu'ara", 227, RevelationPeriod.MIDDLE_MECCAN, 47, "Meccan"),
    27: SurahMetadata(27, "النمل", "The Ant", "An-Naml", 93, RevelationPeriod.MIDDLE_MECCAN, 48, "Meccan"),
    28: SurahMetadata(28, "القصص", "The Stories", "Al-Qasas", 88, RevelationPeriod.MIDDLE_MECCAN, 49, "Meccan"),
    29: SurahMetadata(29, "العنكبوت", "The Spider", "Al-Ankabut", 69, RevelationPeriod.LATE_MECCAN, 85, "Meccan"),
    30: SurahMetadata(30, "الروم", "The Romans", "Ar-Rum", 60, RevelationPeriod.MIDDLE_MECCAN, 84, "Meccan"),
    31: SurahMetadata(31, "لقمان", "Luqman", "Luqman", 34, RevelationPeriod.MIDDLE_MECCAN, 57, "Meccan"),
    32: SurahMetadata(32, "السجدة", "The Prostration", "As-Sajdah", 30, RevelationPeriod.MIDDLE_MECCAN, 75, "Meccan"),
    33: SurahMetadata(33, "الأحزاب", "The Combined Forces", "Al-Ahzab", 73, RevelationPeriod.MEDINAN, 90, "Medinan"),
    34: SurahMetadata(34, "سبأ", "Sheba", "Saba", 54, RevelationPeriod.MIDDLE_MECCAN, 58, "Meccan"),
    35: SurahMetadata(35, "فاطر", "The Originator", "Fatir", 45, RevelationPeriod.MIDDLE_MECCAN, 43, "Meccan"),
    36: SurahMetadata(36, "يس", "Ya-Sin", "Ya-Sin", 83, RevelationPeriod.MIDDLE_MECCAN, 41, "Meccan"),
    37: SurahMetadata(37, "الصافات", "Those Ranged in Ranks", "As-Saffat", 182, RevelationPeriod.MIDDLE_MECCAN, 56, "Meccan"),
    38: SurahMetadata(38, "ص", "Sad", "Sad", 88, RevelationPeriod.MIDDLE_MECCAN, 38, "Meccan"),
    39: SurahMetadata(39, "الزمر", "The Troops", "Az-Zumar", 75, RevelationPeriod.LATE_MECCAN, 59, "Meccan"),
    40: SurahMetadata(40, "غافر", "The Forgiver", "Ghafir", 85, RevelationPeriod.LATE_MECCAN, 60, "Meccan"),
    41: SurahMetadata(41, "فصلت", "Explained in Detail", "Fussilat", 54, RevelationPeriod.LATE_MECCAN, 61, "Meccan"),
    42: SurahMetadata(42, "الشورى", "The Consultation", "Ash-Shura", 53, RevelationPeriod.LATE_MECCAN, 62, "Meccan"),
    43: SurahMetadata(43, "الزخرف", "The Ornaments of Gold", "Az-Zukhruf", 89, RevelationPeriod.LATE_MECCAN, 63, "Meccan"),
    44: SurahMetadata(44, "الدخان", "The Smoke", "Ad-Dukhan", 59, RevelationPeriod.LATE_MECCAN, 64, "Meccan"),
    45: SurahMetadata(45, "الجاثية", "The Crouching", "Al-Jathiyah", 37, RevelationPeriod.LATE_MECCAN, 65, "Meccan"),
    46: SurahMetadata(46, "الأحقاف", "The Wind-Curved Sandhills", "Al-Ahqaf", 35, RevelationPeriod.LATE_MECCAN, 66, "Meccan"),
    47: SurahMetadata(47, "محمد", "Muhammad", "Muhammad", 38, RevelationPeriod.MEDINAN, 95, "Medinan"),
    48: SurahMetadata(48, "الفتح", "The Victory", "Al-Fath", 29, RevelationPeriod.MEDINAN, 111, "Medinan"),
    49: SurahMetadata(49, "الحجرات", "The Rooms", "Al-Hujurat", 18, RevelationPeriod.MEDINAN, 106, "Medinan"),
    50: SurahMetadata(50, "ق", "Qaf", "Qaf", 45, RevelationPeriod.MIDDLE_MECCAN, 34, "Meccan"),
    51: SurahMetadata(51, "الذاريات", "The Winnowing Winds", "Adh-Dhariyat", 60, RevelationPeriod.MIDDLE_MECCAN, 67, "Meccan"),
    52: SurahMetadata(52, "الطور", "The Mount", "At-Tur", 49, RevelationPeriod.EARLY_MECCAN, 76, "Meccan"),
    53: SurahMetadata(53, "النجم", "The Star", "An-Najm", 62, RevelationPeriod.EARLY_MECCAN, 23, "Meccan"),
    54: SurahMetadata(54, "القمر", "The Moon", "Al-Qamar", 55, RevelationPeriod.EARLY_MECCAN, 37, "Meccan"),
    55: SurahMetadata(55, "الرحمن", "The Most Merciful", "Ar-Rahman", 78, RevelationPeriod.EARLY_MECCAN, 97, "Meccan"),
    56: SurahMetadata(56, "الواقعة", "The Inevitable", "Al-Waqi'ah", 96, RevelationPeriod.EARLY_MECCAN, 46, "Meccan"),
    57: SurahMetadata(57, "الحديد", "The Iron", "Al-Hadid", 29, RevelationPeriod.MEDINAN, 94, "Medinan"),
    58: SurahMetadata(58, "المجادلة", "The Pleading Woman", "Al-Mujadilah", 22, RevelationPeriod.MEDINAN, 105, "Medinan"),
    59: SurahMetadata(59, "الحشر", "The Exile", "Al-Hashr", 24, RevelationPeriod.MEDINAN, 101, "Medinan"),
    60: SurahMetadata(60, "الممتحنة", "She That is to be Examined", "Al-Mumtahanah", 13, RevelationPeriod.MEDINAN, 91, "Medinan"),
    61: SurahMetadata(61, "الصف", "The Ranks", "As-Saff", 14, RevelationPeriod.MEDINAN, 109, "Medinan"),
    62: SurahMetadata(62, "الجمعة", "Friday", "Al-Jumu'ah", 11, RevelationPeriod.MEDINAN, 110, "Medinan"),
    63: SurahMetadata(63, "المنافقون", "The Hypocrites", "Al-Munafiqun", 11, RevelationPeriod.MEDINAN, 104, "Medinan"),
    64: SurahMetadata(64, "التغابن", "The Mutual Disillusion", "At-Taghabun", 18, RevelationPeriod.MEDINAN, 108, "Medinan"),
    65: SurahMetadata(65, "الطلاق", "The Divorce", "At-Talaq", 12, RevelationPeriod.MEDINAN, 99, "Medinan"),
    66: SurahMetadata(66, "التحريم", "The Prohibition", "At-Tahrim", 12, RevelationPeriod.MEDINAN, 107, "Medinan"),
    67: SurahMetadata(67, "الملك", "The Sovereignty", "Al-Mulk", 30, RevelationPeriod.MIDDLE_MECCAN, 77, "Meccan"),
    68: SurahMetadata(68, "القلم", "The Pen", "Al-Qalam", 52, RevelationPeriod.EARLY_MECCAN, 2, "Meccan"),
    69: SurahMetadata(69, "الحاقة", "The Reality", "Al-Haqqah", 52, RevelationPeriod.EARLY_MECCAN, 78, "Meccan"),
    70: SurahMetadata(70, "المعارج", "The Ascending Stairways", "Al-Ma'arij", 44, RevelationPeriod.EARLY_MECCAN, 79, "Meccan"),
    71: SurahMetadata(71, "نوح", "Noah", "Nuh", 28, RevelationPeriod.EARLY_MECCAN, 71, "Meccan"),
    72: SurahMetadata(72, "الجن", "The Jinn", "Al-Jinn", 28, RevelationPeriod.EARLY_MECCAN, 40, "Meccan"),
    73: SurahMetadata(73, "المزمل", "The Enshrouded One", "Al-Muzzammil", 20, RevelationPeriod.EARLY_MECCAN, 3, "Meccan"),
    74: SurahMetadata(74, "المدثر", "The Cloaked One", "Al-Muddaththir", 56, RevelationPeriod.EARLY_MECCAN, 4, "Meccan"),
    75: SurahMetadata(75, "القيامة", "The Resurrection", "Al-Qiyamah", 40, RevelationPeriod.EARLY_MECCAN, 31, "Meccan"),
    76: SurahMetadata(76, "الإنسان", "The Human", "Al-Insan", 31, RevelationPeriod.EARLY_MECCAN, 98, "Meccan"),
    77: SurahMetadata(77, "المرسلات", "The Emissaries", "Al-Mursalat", 50, RevelationPeriod.EARLY_MECCAN, 33, "Meccan"),
    78: SurahMetadata(78, "النبأ", "The Tidings", "An-Naba", 40, RevelationPeriod.EARLY_MECCAN, 80, "Meccan"),
    79: SurahMetadata(79, "النازعات", "Those Who Drag Forth", "An-Nazi'at", 46, RevelationPeriod.EARLY_MECCAN, 81, "Meccan"),
    80: SurahMetadata(80, "عبس", "He Frowned", "Abasa", 42, RevelationPeriod.EARLY_MECCAN, 24, "Meccan"),
    81: SurahMetadata(81, "التكوير", "The Overthrowing", "At-Takwir", 29, RevelationPeriod.EARLY_MECCAN, 7, "Meccan"),
    82: SurahMetadata(82, "الانفطار", "The Cleaving", "Al-Infitar", 19, RevelationPeriod.EARLY_MECCAN, 82, "Meccan"),
    83: SurahMetadata(83, "المطففين", "The Defrauding", "Al-Mutaffifin", 36, RevelationPeriod.EARLY_MECCAN, 86, "Meccan"),
    84: SurahMetadata(84, "الانشقاق", "The Sundering", "Al-Inshiqaq", 25, RevelationPeriod.EARLY_MECCAN, 83, "Meccan"),
    85: SurahMetadata(85, "البروج", "The Mansions of the Stars", "Al-Buruj", 22, RevelationPeriod.EARLY_MECCAN, 27, "Meccan"),
    86: SurahMetadata(86, "الطارق", "The Morning Star", "At-Tariq", 17, RevelationPeriod.EARLY_MECCAN, 36, "Meccan"),
    87: SurahMetadata(87, "الأعلى", "The Most High", "Al-A'la", 19, RevelationPeriod.EARLY_MECCAN, 8, "Meccan"),
    88: SurahMetadata(88, "الغاشية", "The Overwhelming", "Al-Ghashiyah", 26, RevelationPeriod.EARLY_MECCAN, 68, "Meccan"),
    89: SurahMetadata(89, "الفجر", "The Dawn", "Al-Fajr", 30, RevelationPeriod.EARLY_MECCAN, 10, "Meccan"),
    90: SurahMetadata(90, "البلد", "The City", "Al-Balad", 20, RevelationPeriod.EARLY_MECCAN, 35, "Meccan"),
    91: SurahMetadata(91, "الشمس", "The Sun", "Ash-Shams", 15, RevelationPeriod.EARLY_MECCAN, 26, "Meccan"),
    92: SurahMetadata(92, "الليل", "The Night", "Al-Layl", 21, RevelationPeriod.EARLY_MECCAN, 9, "Meccan"),
    93: SurahMetadata(93, "الضحى", "The Morning Hours", "Ad-Duha", 11, RevelationPeriod.EARLY_MECCAN, 11, "Meccan"),
    94: SurahMetadata(94, "الشرح", "The Relief", "Ash-Sharh", 8, RevelationPeriod.EARLY_MECCAN, 12, "Meccan"),
    95: SurahMetadata(95, "التين", "The Fig", "At-Tin", 8, RevelationPeriod.EARLY_MECCAN, 28, "Meccan"),
    96: SurahMetadata(96, "العلق", "The Clot", "Al-Alaq", 19, RevelationPeriod.EARLY_MECCAN, 1, "Meccan"),
    97: SurahMetadata(97, "القدر", "The Power", "Al-Qadr", 5, RevelationPeriod.EARLY_MECCAN, 25, "Meccan"),
    98: SurahMetadata(98, "البينة", "The Clear Proof", "Al-Bayyinah", 8, RevelationPeriod.MEDINAN, 100, "Medinan"),
    99: SurahMetadata(99, "الزلزلة", "The Earthquake", "Az-Zalzalah", 8, RevelationPeriod.MEDINAN, 93, "Medinan"),
    100: SurahMetadata(100, "العاديات", "The Chargers", "Al-Adiyat", 11, RevelationPeriod.EARLY_MECCAN, 14, "Meccan"),
    101: SurahMetadata(101, "القارعة", "The Calamity", "Al-Qari'ah", 11, RevelationPeriod.EARLY_MECCAN, 30, "Meccan"),
    102: SurahMetadata(102, "التكاثر", "The Rivalry in Worldly Increase", "At-Takathur", 8, RevelationPeriod.EARLY_MECCAN, 16, "Meccan"),
    103: SurahMetadata(103, "العصر", "The Declining Day", "Al-Asr", 3, RevelationPeriod.EARLY_MECCAN, 13, "Meccan"),
    104: SurahMetadata(104, "الهمزة", "The Traducer", "Al-Humazah", 9, RevelationPeriod.EARLY_MECCAN, 32, "Meccan"),
    105: SurahMetadata(105, "الفيل", "The Elephant", "Al-Fil", 5, RevelationPeriod.EARLY_MECCAN, 19, "Meccan"),
    106: SurahMetadata(106, "قريش", "Quraysh", "Quraysh", 4, RevelationPeriod.EARLY_MECCAN, 29, "Meccan"),
    107: SurahMetadata(107, "الماعون", "The Small Kindnesses", "Al-Ma'un", 7, RevelationPeriod.EARLY_MECCAN, 17, "Meccan"),
    108: SurahMetadata(108, "الكوثر", "The Abundance", "Al-Kawthar", 3, RevelationPeriod.EARLY_MECCAN, 15, "Meccan"),
    109: SurahMetadata(109, "الكافرون", "The Disbelievers", "Al-Kafirun", 6, RevelationPeriod.EARLY_MECCAN, 18, "Meccan"),
    110: SurahMetadata(110, "النصر", "The Divine Support", "An-Nasr", 3, RevelationPeriod.MEDINAN, 114, "Medinan"),
    111: SurahMetadata(111, "المسد", "The Palm Fibre", "Al-Masad", 5, RevelationPeriod.EARLY_MECCAN, 6, "Meccan"),
    112: SurahMetadata(112, "الإخلاص", "The Sincerity", "Al-Ikhlas", 4, RevelationPeriod.EARLY_MECCAN, 22, "Meccan"),
    113: SurahMetadata(113, "الفلق", "The Daybreak", "Al-Falaq", 5, RevelationPeriod.EARLY_MECCAN, 20, "Meccan"),
    114: SurahMetadata(114, "الناس", "Mankind", "An-Nas", 6, RevelationPeriod.EARLY_MECCAN, 21, "Meccan"),
}


def get_surahs_by_period(period: RevelationPeriod) -> List[SurahMetadata]:
    """Return all surahs belonging to a specific revelation period."""
    return [s for s in SURAH_METADATA.values() if s.revelation_period == period]


def get_period_distribution() -> Dict[str, int]:
    """Return count of surahs per revelation period."""
    dist = {}
    for period in RevelationPeriod:
        count = len(get_surahs_by_period(period))
        dist[period.value] = count
    return dist


def get_chronological_order() -> List[SurahMetadata]:
    """Return surahs in chronological revelation order."""
    return sorted(SURAH_METADATA.values(), key=lambda s: s.chronological_order)


def validate_metadata():
    """Validate the metadata for completeness and consistency."""
    assert len(SURAH_METADATA) == 114, f"Expected 114 surahs, got {len(SURAH_METADATA)}"

    total_verses = sum(s.verse_count for s in SURAH_METADATA.values())
    assert total_verses == 6236, f"Expected 6236 total verses, got {total_verses}"

    # Verify chronological orders are unique and complete (1-114)
    chrono_orders = sorted(s.chronological_order for s in SURAH_METADATA.values())
    assert chrono_orders == list(range(1, 115)), "Chronological orders must be 1-114 unique"

    # Verify period distribution
    dist = get_period_distribution()
    print(f"Revelation period distribution:")
    for period, count in dist.items():
        print(f"  {period}: {count} surahs")
    print(f"  Total: {sum(dist.values())} surahs")
    print(f"  Total verses: {total_verses}")
    print("✓ Metadata validation passed")


if __name__ == "__main__":
    validate_metadata()
