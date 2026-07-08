"""Builds the multilingual prompt sent to the LLM Provider.

The Summary is produced directly in Markdown by the LLM, between sentinels that
allow robust extraction even when the CLI adds banners/ANSI/extra text.
"""

from __future__ import annotations

# Sentinels that delimit the Markdown of the Summary in the LLM output.
BEGIN_MARK = "===NOTETAKER-SUMMARY-START==="
END_MARK = "===NOTETAKER-SUMMARY-END==="

# Target language name by code, to instruct the LLM.
_LANG_NAME = {
    "pt": "Brazilian Portuguese",
    "es": "Spanish",
    "en": "English",
}

# Section titles by language (the LLM must use exactly these).
_SECTIONS = {
    "pt": ["Resumo Executivo", "Pontos Discutidos", "Decisoes", "Tarefas", "Observacoes"],
    "es": ["Resumen Ejecutivo", "Puntos Discutidos", "Decisiones", "Tareas", "Observaciones"],
    "en": ["Executive Summary", "Discussion Points", "Decisions", "Action Items", "Notes"],
}


def resolve_output_language(output_lang: str, meeting_lang: str) -> str:
    """Resolve the output language of the Summary.

    output_lang can be 'meeting' (follows the Meeting Language) or pt/es/en.
    """
    if output_lang and output_lang != "meeting":
        return output_lang
    if meeting_lang in _LANG_NAME:
        return meeting_lang
    return "pt"


def build_prompt(transcript: str, output_lang: str, title: str = "") -> str:
    """Build the complete prompt: instructions + Markdown structure + transcript."""
    lang = output_lang if output_lang in _SECTIONS else "pt"
    lang_name = _LANG_NAME[lang]
    s = _SECTIONS[lang]
    heading = title or {"pt": "Resumo da Reuniao", "es": "Resumen de la Reunion",
                        "en": "Meeting Summary"}[lang]

    return f"""\
You are an assistant that summarizes meetings from a transcript.

The transcript is labeled by speaker (e.g., [You], [Participants]). Use
these labels to assign tasks to the correct responsible person.

Generate a structured summary in Markdown, WRITTEN IN {lang_name.upper()}.

Write the Markdown between the two sentinel lines below, with no other text
before or after them:

{BEGIN_MARK}
# {heading}

## {s[0]}
(2 to 3 sentences summarizing the meeting)

## {s[1]}
(list of objective topics, one per line with "- ")

## {s[2]}
(only what was effectively decided; "- " per item)

## {s[3]}
(each concrete action as "- description (responsible person; deadline)"; use "-" when there is no deadline)

## {s[4]}
(risks, open questions, items to track; "- " per item)
{END_MARK}

Rules:
- Use exactly these section titles and in this order.
- If a section has no content, write "- (none)".
- Do not include code blocks, only text Markdown.

Meeting transcript:
---
{transcript}
---
"""
