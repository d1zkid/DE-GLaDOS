# ruff: noqa: RUF001, RUF002
import re
from typing import ClassVar


class SpokenTextConverter:
    """
    Eine Hilfsklasse zur Umwandlung von Text mit Zahlen, Daten, Uhrzeiten und Währungen
    in ihre gesprochenen deutschen Entsprechungen.

    Beispiel:
        >>> converter = SpokenTextConverter()
        >>> result = converter.text_to_spoken("Das Meeting ist um 15:00 Uhr am 1.1.2024.")
        >>> print(result)
        Das Meeting ist um fünfzehn Uhr am ersten ersten zweitausendvierundzwanzig.
    """

    # German contractions / informal shortenings
    CONTRACTIONS: ClassVar[dict[str, str]] = {
        "ich'm": "ich bin",
        "ich'll": "ich werde",
        "ich've": "ich habe",
        "ich'd": "ich würde",
        "nicht": "nicht",
        "kann't": "kann nicht",
        "won't": "wird nicht",
        "n't": " nicht",
        "'ll": " wird",
        "'re": " sind",
        "'ve": " haben",
        "'m": " bin",
        "'d": " würde",
    }

    def __init__(self) -> None:
        self.convertible_pattern = re.compile(
            r"""(?x)
            \d                          # Jede Ziffer
            |\$|£|€                     # Währungssymbole
            |[×÷^√∛]                   # Mathematische Operatoren
            |\b(?:Dr|Hr|Fr|Prof)\.     # Deutsche Abkürzungen
            |\.{3,}|\. \. \.           # Auslassungspunkte
            """
        )

    def _number_to_words(self, num: float | str) -> str:
        """
        Wandelt eine Zahl in ihre deutsche Wortform um.

        Beispiele:
            42      → 'zweiundvierzig'
            -17     → 'minus siebzehn'
            1000000 → 'eine Million'
            3.14    → 'drei Komma eins vier'
        """
        try:
            if isinstance(num, str):
                if "." not in num or num.endswith(".0"):
                    num = int(float(num))
                else:
                    num = float(num)

            if isinstance(num, float) and num.is_integer():
                num = int(num)

            if num == 0:
                return "null"

            ones = [
                "null", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
                "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
                "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
            ]
            # Standalone forms (used when the number stands alone, not in compounds)
            ones_standalone = [
                "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben",
                "acht", "neun", "zehn", "elf", "zwölf", "dreizehn", "vierzehn",
                "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn",
            ]
            tens = [
                "", "", "zwanzig", "dreißig", "vierzig", "fünfzig",
                "sechzig", "siebzig", "achtzig", "neunzig",
            ]

            def process_chunk(n: int, scale: int, standalone: bool = False) -> str:
                """Verarbeitet einen dreistelligen Zahlenblock."""
                if n == 0:
                    return ""

                hundreds = n // 100
                remainder = n % 100
                words = []

                if hundreds > 0:
                    words.append(f"{ones[hundreds]}hundert")

                if remainder > 0:
                    if remainder < 20:
                        # Use standalone form only for top-level single numbers
                        word = ones_standalone[remainder] if (standalone and scale == 0 and hundreds == 0) else ones[remainder]
                        words.append(word)
                    else:
                        tens_digit = remainder // 10
                        ones_digit = remainder % 10
                        if ones_digit == 0:
                            words.append(tens[tens_digit])
                        else:
                            # German: einundzwanzig, zweiunddreißig, etc.
                            words.append(f"{ones[ones_digit]}und{tens[tens_digit]}")

                result = "".join(words)

                # Scale words in German
                if scale == 1 and result:
                    result += "tausend"
                elif scale == 2 and result:
                    # Million is separate and uses spaces
                    if n == 1:
                        result = "eine Million "
                    else:
                        result = result + " Millionen "
                elif scale == 3 and result:
                    if n == 1:
                        result = "eine Milliarde "
                    else:
                        result = result + " Milliarden "

                return result

            if num < 0:
                return "minus " + self._number_to_words(abs(num))

            if isinstance(num, int):
                if num == 0:
                    return "null"

                # Special case for exactly 1 standalone
                if num == 1:
                    return "eins"

                result_parts: list[str] = []
                scale = 0
                remaining = num

                while remaining > 0:
                    chunk = remaining % 1000
                    if chunk != 0:
                        chunk_words = process_chunk(chunk, scale, standalone=(num < 20))
                        result_parts.insert(0, chunk_words)
                    remaining //= 1000
                    scale += 1

                return "".join(filter(None, result_parts))
            else:
                # Decimal numbers
                str_num = f"{num:.10f}".rstrip("0")
                if "." in str_num:
                    int_part, dec_part = str_num.split(".")
                else:
                    int_part, dec_part = str_num, ""

                int_num = int(int_part)

                if int_num == 0:
                    result = "null"
                elif int_num == 1:
                    result = "eins"
                else:
                    result_parts = []
                    scale = 0
                    while int_num > 0:
                        chunk = int_num % 1000
                        if chunk != 0:
                            chunk_words = process_chunk(chunk, scale)
                            result_parts.insert(0, chunk_words)
                        int_num //= 1000
                        scale += 1
                    result = "".join(filter(None, result_parts))

                if dec_part:
                    # German decimal separator is "Komma"
                    digit_words = [ones_standalone[int(d)] for d in dec_part]
                    result = result + " Komma " + " ".join(digit_words)

                return result

        except (ValueError, TypeError) as e:
            raise ValueError(f"Ungültiges Zahlenformat: {num}") from e

    def _split_num(self, num: re.Match) -> str:
        """
        Wandelt Uhrzeiten und Jahreszahlen in ihre deutsche Lautform um.

        Uhrzeiten: 15:00 → 'fünfzehn Uhr'
        Jahre:     1999  → 'neunzehnhundertneunundneunzig'
        """
        try:
            match_str = num.group()

            if ":" in match_str:
                time_str = match_str.lower()

                try:
                    h, m = [int(n) for n in time_str.split(":")]
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        return match_str

                    if m == 0:
                        return f"{self._number_to_words(h)} Uhr"
                    elif m < 10:
                        return f"{self._number_to_words(h)} Uhr null{self._number_to_words(m)}"
                    else:
                        return f"{self._number_to_words(h)} Uhr {self._number_to_words(m)}"
                except ValueError:
                    return match_str

            # Year handling
            try:
                number = int(match_str.rstrip("er"))  # Handle "1990er"
                is_decade = match_str.endswith("er")

                if len(match_str.rstrip("er")) == 4:
                    left, right = divmod(number, 100)

                    if number == 2000:
                        return "zweitausend" + ("er" if is_decade else "")
                    elif number >= 2001:
                        # 2001–2099: zweitausend[rest]
                        rest = self._number_to_words(right) if right else ""
                        base = "zweitausend" + rest
                    elif right == 0:
                        base = f"{self._number_to_words(left)}hundert"
                    elif right < 10:
                        base = f"{self._number_to_words(left)}hundert{self._number_to_words(right)}"
                    else:
                        base = f"{self._number_to_words(left)}hundert{self._number_to_words(right)}"

                    return base + ("er" if is_decade else "")

                return self._number_to_words(number)
            except ValueError:
                return match_str
        except Exception:
            return num.group()

    def _flip_money(self, m: re.Match[str]) -> str:
        """
        Wandelt Währungsangaben in ihre deutsche Lautform um.

        "$5.00"  → 'fünf Dollar'
        "€10.50" → 'zehn Euro und fünfzig Cent'
        "£1.00"  → 'ein Pfund'
        """
        try:
            text = m.group()
            if not text or len(text) < 2:
                raise ValueError("Ungültiges Währungsformat")

            symbol = text[0]
            if symbol == "$":
                main_unit = "Dollar"
                sub_unit_singular = "Cent"
                sub_unit_plural = "Cent"
            elif symbol == "€":
                main_unit = "Euro"
                sub_unit_singular = "Cent"
                sub_unit_plural = "Cent"
            else:  # £
                main_unit = "Pfund"
                sub_unit_singular = "Penny"
                sub_unit_plural = "Pence"

            amount_str = text[1:]

            if "." not in amount_str:
                amount = int(amount_str)
                return f"{self._number_to_words(amount)} {main_unit}"

            b, c = amount_str.split(".")
            if not b:
                b = "0"
            c_int = int(c.ljust(2, "0"))

            main_str = self._number_to_words(int(b))

            if c_int == 0:
                return f"{main_str} {main_unit}"

            sub_unit = sub_unit_singular if c_int == 1 else sub_unit_plural
            return f"{main_str} {main_unit} und {self._number_to_words(c_int)} {sub_unit}"
        except Exception:
            return m.group()

    def _point_num(self, num: re.Match[str]) -> str:
        """Wandelt eine Dezimalzahl in ihre deutsche Lautform um."""
        return self._number_to_words(float(num.group()))

    def _convert_percentages(self, text: str) -> str:
        """
        Wandelt Prozentzahlen in ihre deutsche Lautform um.

        '25%'   → 'fünfundzwanzig Prozent'
        '99.5%' → 'neunundneunzig Komma fünf Prozent'
        """
        def replace_match(match: re.Match) -> str:
            number = match.group(1)
            if "." not in number:
                return f"{self._number_to_words(int(number))} Prozent"
            return f"{self._number_to_words(float(number))} Prozent"

        return re.sub(r"(\d+\.?\d*)%", replace_match, text)

    def _contains_convertible_content(self, text: str) -> bool:
        """Schnelle Prüfung ob der Text umwandelbaren Inhalt enthält."""
        return bool(self.convertible_pattern.search(text))

    def _convert_mathematical_notation(self, text: str) -> str:
        """
        Wandelt mathematische Notation in ihre deutsche Lautform um.

        '8^2'  → 'acht hoch zwei'
        '√9'   → 'Wurzel aus neun'
        '1/2'  → 'ein halb'  (or 'eins durch zwei' for arbitrary fractions)
        """
        if not self._contains_convertible_content(text):
            return text

        def convert_numbers_in_match(match: re.Match, pattern: str) -> str:
            parts = list(match.groups())
            for i, part in enumerate(parts):
                if part and part.isdigit():
                    parts[i] = self._number_to_words(int(part))
            return pattern.format(*parts)

        # Basic arithmetic symbols — German spoken forms
        text = text.replace(" = ", " gleich ")
        text = text.replace("=", " gleich ")
        text = text.replace(" + ", " plus ")
        text = text.replace("+", " plus ")
        text = text.replace(" - ", " minus ")
        text = text.replace(" × ", " mal ")
        text = text.replace("×", " mal ")
        text = text.replace(" ÷ ", " geteilt durch ")
        text = text.replace("÷", " geteilt durch ")

        # Exponents: 8^2 → acht hoch zwei
        text = re.sub(
            r"(\d+)\^(\d+)",
            lambda m: convert_numbers_in_match(m, "{0} hoch {1}"),
            text,
        )

        # Letter variables with exponents: x^2 → x hoch zwei
        text = re.sub(
            r"([a-zA-Z])\^(\d+)",
            lambda m: f"{m.group(1)} hoch {self._number_to_words(int(m.group(2)))}",
            text,
        )

        # Square roots: √9 → Wurzel aus neun
        text = re.sub(
            r"√(\d+)",
            lambda m: f"Wurzel aus {self._number_to_words(int(m.group(1)))}",
            text,
        )

        # Cube roots: ∛8 → dritte Wurzel aus acht
        text = re.sub(
            r"∛(\d+)",
            lambda m: f"dritte Wurzel aus {self._number_to_words(int(m.group(1)))}",
            text,
        )

        # Fractions (skip dates)
        def convert_fraction(match: re.Match) -> str:
            if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", match.group(0)):
                return match.group(0)
            num = self._number_to_words(int(match.group(1)))
            den = self._number_to_words(int(match.group(2)))
            return f"{num} durch {den}"

        text = re.sub(r"(\d+)/(\d+)(?!/)", convert_fraction, text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def text_to_spoken(self, text: str) -> str:
        """
        Wandelt einen Text in seine gesprochene deutsche Form um.

        Verarbeitet:
        - Abkürzungen (Dr., Hr., Fr., Prof.)
        - Zahlen (27 → siebenundzwanzig)
        - Uhrzeiten (15:00 → fünfzehn Uhr)
        - Währungen (€10,50 → zehn Euro und fünfzig Cent)
        - Prozentzahlen (25% → fünfundzwanzig Prozent)
        - Mathematische Notation
        - Jahreszahlen (1999 → neunzehnhundertneunundneunzig)
        """
        # 1. Expand contractions
        for contraction, expansion in sorted(self.CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(contraction, expansion)

        # Remove leading/trailing whitespace per line
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        # 2. Quote normalization
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        text = text.replace("(", "«").replace(")", "»")

        # 3. Punctuation normalization
        for a, b in zip("、。！，：；？", ",.!,:;?", strict=False):
            text = text.replace(a, b + " ")

        # Remove ellipses
        text = re.sub(r"\.{3,}|\. \. \.", "", text)

        # 4. Whitespace normalization
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)

        # 5. German titles and abbreviations
        text = re.sub(r"\bDr\.(?= [A-ZÄÖÜ])", "Doktor", text)
        text = re.sub(r"\bProf\.(?= [A-ZÄÖÜ])", "Professor", text)
        text = re.sub(r"\bHr\.(?= [A-ZÄÖÜ])", "Herr", text)
        text = re.sub(r"\bFr\.(?= [A-ZÄÖÜ])", "Frau", text)
        text = re.sub(r"\busw\.(?! [A-ZÄÖÜ])", "und so weiter", text)
        text = re.sub(r"\bzB\.?(?! [A-ZÄÖÜ])", "zum Beispiel", text)
        text = re.sub(r"\bca\.(?! [A-ZÄÖÜ])", "circa", text)

        # Split acronyms into individual letters (but preserve German words)
        def process_word(match: re.Match) -> str:
            word = match.group(0)
            if word.isupper() and len(word) > 1:
                return " ".join(word)
            return word

        text = re.sub(r"\b[A-ZÄÖÜ]{2,}\b", process_word, text)

        # 6. Large numbers with thousands separators (German uses . as separator)
        def preserve_large_numbers(match: re.Match) -> str:
            num = int(match.group().replace(".", "").replace(",", ""))
            return self._number_to_words(num)

        # Handle German-style 1.000.000 and English-style 1,000,000
        text = re.sub(r"\b\d{1,3}(?:\.\d{3})+\b", preserve_large_numbers, text)
        text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", preserve_large_numbers, text)

        # 7. Date conversion — German format DD.MM.YYYY
        def convert_date(match: re.Match) -> str:
            parts = match.group().split(".")
            if len(parts) == 3 and len(parts[2]) == 4:
                year = int(parts[2])
                if year == 2000:
                    year_text = "zweitausend"
                elif year >= 2001:
                    left, right = divmod(year, 100)
                    year_text = "zweitausend" + (self._number_to_words(right) if right else "")
                else:
                    left, right = divmod(year, 100)
                    if right == 0:
                        year_text = f"{self._number_to_words(left)}hundert"
                    else:
                        year_text = f"{self._number_to_words(left)}hundert{self._number_to_words(right)}"
                return (
                    f"{self._number_to_words(int(parts[0]))}ten "
                    f"{self._number_to_words(int(parts[1]))}ten "
                    f"{year_text}"
                )
            return ".".join(self._number_to_words(int(part)) for part in parts if part)

        # German date format: DD.MM.YYYY
        text = re.sub(r"\b\d{1,2}\.\d{1,2}\.(?:\d{4}|\d{2})\b", convert_date, text)

        # Also handle slash-separated dates
        def convert_slash_date(match: re.Match) -> str:
            parts = match.group().split("/")
            return "/".join(self._number_to_words(int(p)) for p in parts)

        text = re.sub(r"\b\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})\b", convert_slash_date, text)

        # 8. Mathematical notation
        text = self._convert_mathematical_notation(text)

        # 9. Number conversions in order:
        # a. Percentages
        text = self._convert_percentages(text)

        # b. Currency (€, $, £)
        text = re.sub(
            r"(?i)[€$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[€$£]\d+[.,]\d\d?\b",
            self._flip_money,
            text,
        )

        # c. Times (24h format common in German)
        text = re.sub(r"\b(\d{1,2}):(\d{2})\b", self._split_num, text)

        # d. Years (4-digit, optionally followed by "er" for decades like "1990er")
        text = re.sub(
            r"\b\d{4}(?:er)?\b",
            lambda m: self._split_num(m),
            text,
        )

        # e. Decimal numbers (German uses comma, but handle dot too)
        text = re.sub(r"\d*\.\d+", self._point_num, text)

        # f. Standalone integers
        text = re.sub(r"\b\d+\b", lambda m: self._number_to_words(int(m.group())), text)

        # 10. Final formatting
        text = re.sub(r"(?<=\d)-(?=\d)", " bis ", text)   # ranges: 3-5 → drei bis fünf
        text = re.sub(r"  +", " ", text)

        return text.strip()