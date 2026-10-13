import re


_GREETING_WORDS = {"hi", "hello", "hey", "hii"}
_ACKNOWLEDGEMENTS = {"thanks", "thank you", "thx"}
_NEUTRAL_ACKNOWLEDGEMENTS = {"ok", "okay"}
_DAY_GREETINGS = {"good morning", "good afternoon", "good evening"}
_CANNED_REPLIES = {
    "hi how can i help you today",
    "you re welcome how can i help",
}


def _looks_like_name(words: list[str]) -> bool:
    if not 1 <= len(words) <= 3:
        return False

    return all(
        re.fullmatch(r"[A-Z][A-Za-z'-]*", word) is not None
        for word in words
    )


def is_trivial_small_talk(text: str) -> bool:
    content = text.strip()
    if not content:
        return True

    words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", content)
    if not words:
        return False

    normalized_words = [word.lower() for word in words]
    normalized = " ".join(normalized_words)

    if (
        normalized in _ACKNOWLEDGEMENTS
        or normalized in _NEUTRAL_ACKNOWLEDGEMENTS
        or normalized in _DAY_GREETINGS
        or normalized in _CANNED_REPLIES
    ):
        return True

    if normalized_words[0] in _GREETING_WORDS:
        if len(words) == 1:
            return True

        if len(words) == 2:
            return _looks_like_name(words[1:])

        if (
            len(words) >= 4
            and normalized_words[1:3] == ["i", "am"]
            and _looks_like_name(words[3:])
        ):
            return True

        if (
            len(words) >= 3
            and normalized_words[1] == "i'm"
            and _looks_like_name(words[2:])
        ):
            return True

        if (
            len(words) >= 5
            and normalized_words[1:4] == ["my", "name", "is"]
            and _looks_like_name(words[4:])
        ):
            return True

        return False

    return False


def small_talk_reply(text: str) -> str:
    normalized = " ".join(
        word.lower()
        for word in re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", text)
    )
    if normalized in _ACKNOWLEDGEMENTS:
        return "You're welcome! How can I help?"
    if normalized in _NEUTRAL_ACKNOWLEDGEMENTS:
        return "Okay."
    return "Hi! How can I help you today?"
