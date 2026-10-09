# Ingestion

`uv run dpdp-ingest` reads the PDFs listed in `configs/ingest.yaml` and writes:

- `data/processed/chunks.jsonl`, one JSON object per chunk
- `data/processed/corrigendum_patches.json`, the parsed G.S.R. 892(E) patches and where they landed

## Pipeline

1. **`pdf.py`** reads each page with pymupdf at span level and rebuilds *visual* lines by
   grouping spans on the same baseline. This matters for two reasons:
   - The corrigendum counts printed lines. Pymupdf's plain-text mode would split "(i)" and
     its text into two lines.
   - Line numbers count body lines below the running header. This matches the Gazette:
     page 24 line 22 is rule 1(3), and page 38 line 2 is "2019 (35 of 2019).".

   The following are removed at this stage:
   - running headers and footers, from per-document y-bands; the printed page number is
     read from them
   - the India Code watermark: rotated or grey spans, plus stand-alone lines that are 1-2
     letter fragments of "IndiaCode" ("aC", "di", "In")
   - superscript footnote markers
   - India Code footnotes below the short separator rule
   - "Uploaded by …" footers

   Tables are taken from `page.find_tables()`, for Rules Schedules 3, 4 and 7.
2. **`act.py`** keeps only the text from the long title to the Statement of Objects and
   Reasons. This drops the cover, the abbreviations page and the table of contents
   ("Arrangement of Sections"). It then segments:
   - `CHAPTER` lines set the `chapter` and `chapter_title` metadata.
   - A section starts at "N. Title.—", accepted only if N is the next section number.
   - A sub-section starts at "(n)", accepted only if n is the next number and the previous
     line ends a sentence.
   - Illustrations and Explanations stay with their sub-section.
   - A sub-section is split into clause chunks only when it is longer than
     `max_chunk_chars` (s. 2, s. 7, s. 40(2)) or when G.S.R. 843(E) gives its clauses
     different dates (s. 27(1)(d)). Clause chunks keep the lead-in sentence in `lead_in`.

   The penalty Schedule has no ruling that pymupdf detects. It is read column by column
   (`schedule_penalty_column_x`) and gives one chunk per row. Each row's penalty is also
   copied onto the section chunks it cites (`penalty`).
3. **`rules.py`** produces:
   - one chunk per rule, or per sub-rule when a rule has sub-rules
   - one chunk per Illustration, with all its Cases kept together and listed in `cases`
   - one chunk per Schedule row: numbered paragraphs (First, Fifth and Sixth), lettered
     clauses (Second) or table rows (Third, Fourth and Seventh). Table rows that continue
     across a page are merged.
   - one chunk per clause of a Schedule `Note`

   Schedule rows carry `schedule`, `part` and `item`.
4. **`notifications.py`** produces one chunk per lettered clause or numbered paragraph,
   plus the opening "In exercise of the powers…" text.
5. **`corrigendum.py`** parses G.S.R. 892(E) into
   `{page, line, line_end, old_text, new_text, kind, target_rule_or_schedule, ...}`.
   - **Matching.** Each patch is searched for by its quoted `old_text`, using word
     boundaries, among the chunks on the cited printed page. If `old_text` occurs once on
     that page, that occurrence is used. If it occurs more than once ("of this Gazette" on
     page 24, "." on page 38), the occurrence nearest the cited line wins. The method is
     recorded in `match_strategy`.
   - **Relettering.** Patch (v)(b), "lines 1 to 15, for (a) to (f), read (a) to (g)",
     re-letters a list rather than replacing text. The Fourth Schedule Note was printed
     with two clauses labelled (a). The patch is applied only if the printed labels are
     exactly (a)-(f) over seven clauses, and they are re-lettered (a)-(g).
   - **What a corrected chunk keeps.** `text_original` and `text_corrected` (equal to
     `text`), and `corrected_by: "G.S.R. 892(E)"`.
   - Each patch also becomes a `doc_type: corrigendum` chunk that points at its target
     rule or schedule.
6. **`in_force.py`** parses commencement from the documents themselves:
   - the publication date, from "New Delhi, the 13th November, 2025"
   - the offset ("one year", "eighteen months")
   - the provision lists, using the reference parser in `refs.py`

   Rules 1, 2 and 17-21 → 2025-11-13. Rule 4 → 2026-11-13. Rules 3, 5-16, 22 and 23 →
   2027-05-13. For the Act, the most specific listed provision wins (s. 27 vs s. 27(1)(d)).
   Schedules take the earliest date of the rules in their `[See rule …]` line. The Act's
   Schedule takes the date of s. 33.
7. **`refs.py`** fills `refers_to` (`["section 9"]`) and `refers_to_provisions`
   (`["section 9(1)", "section 9(3)"]`).
   - It drops sections of other enactments, e.g. "section 15 of the Rights of Persons with
     Disabilities Act" or "section 21 of the Indian Penal Code".
   - Schedule rows also inherit references made in their Schedule or Part heading.

## Chunk schema

| Field | Notes |
|---|---|
| `chunk_id` | Stable id, e.g. `dpdp_act_2023:s6(9)`, `dpdp_act_2023:s27(1)(d)`, `dpdp_rules_2025:r8:ill`, `dpdp_rules_2025:sch4:A:3`, `dpdp_rules_2025:sch4:note2`, `gsr_843e:b`, `gsr_892e:(v)(b)` |
| `doc_id`, `doc_type`, `gsr_no`, `source_file` | `doc_type` ∈ act, rules, notification, corrigendum |
| `title` | Section/rule heading, or Schedule (and Part) title |
| `chapter`, `chapter_title` | Act only |
| `section`, `sub_section`, `clause`, `lead_in` | Act; `clause` also labels notification clauses |
| `rule`, `sub_rule` | Rules |
| `schedule`, `part`, `item` | Schedule rows; `item` is "3" or "Note (b)" |
| `illustration`, `cases` | Illustration chunks and their Case labels |
| `page`, `page_end` | Printed page numbers (Rules: 24-41) |
| `text` | Current (corrected) text, exactly as printed otherwise |
| `in_force_date` | ISO date |
| `refers_to`, `refers_to_provisions` | DPDP Act sections referenced |
| `penalty` | Act: Schedule penalty for the provision |
| `text_original`, `text_corrected`, `corrected_by` | Set only on chunks changed by G.S.R. 892(E) |

## Judgement calls

- **Section 1(1)** (short title) is not named in G.S.R. 843(E). It and the long title take
  the date of assent, 2023-08-11 (`enactment_date` in the config). The pipeline logs a
  warning for this.
- **India Code footnote.** The footnote to s. 1 disagrees with G.S.R. 843(E) on ss. 36-37.
  The notification is followed.
- **Notifications' own dates.** Notification chunks carry their publication date (the date
  they took effect), not the dates they assign to the Act.
- **Not indexed.** The Statement of Objects and Reasons (India Code p. 25) is not part of
  the Act and is not indexed.
- **Rules split per rule, not per length.** `max_chunk_chars` applies only to the Act. The
  longest Rules chunk is the rule 10 Illustration (~2.3k characters).
