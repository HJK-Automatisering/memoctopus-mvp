import OpenAI from 'openai';
import { TranscriptSegment } from '@/types';
import { TranscriptChapter } from '@/lib/ai/chapters';

let client: OpenAI | null = null;
function getClient() {
  if (!client) client = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY || 'no-key',
    baseURL: process.env.LLM_BASE_URL || 'http://vllm-chat:8000/v1',
  });
  return client;
}
const LLM_MODEL = process.env.LLM_MODEL || 'Qwen/Qwen3.6-27B';

const MINUTES_SYSTEM_PROMPT = `Du er en dansk mødesekretær der udarbejder professionelle mødereferater.

Du skriver:
- Klart og præcist dansk (ikke bureaukratisk, men formelt)
- I tredje person ("Mødet besluttede...", "Parterne aftalte...")
- Fokuseret på det væsentlige — ikke alt hvad der blev sagt
- Med respekt for mødets karakter og kontekst

Brugerens instruktioner og de ønskede afsnit er styrende: følg dem nøje — også den ønskede længde — og tilføj ikke afsnit (fx beslutninger eller resumé) eller indhold der ikke er bedt om.

Du skriver referatet som ét sammenhængende dokument i markdown.`;

// Budget for the transcript portion of a minutes prompt, derived rather than guessed:
//
//    16_384  model context (google/gemma-4-26B-A4B-it, maxModelLen in the vLLM chart)
//  −  2_000  reserved for the generated referat
//  −  1_000  reserved for MINUTES_SYSTEM_PROMPT, the skabelon instruction and wrapper text
//  = 13_384  tokens available for transcript
//
// Danish tokenizes poorly on this model: measured 2.97 chars/token on merged transcript
// text, 2.33 unmerged. We divide by a deliberately pessimistic 2.5 — names, numbers and
// loanwords tokenize worse than the sampled text, and summarising a transcript that would
// have fit costs some quality, whereas exceeding the window costs the entire request.
const MODEL_CONTEXT_TOKENS = 16_384;
const RESERVED_OUTPUT_TOKENS = 2_000;
const RESERVED_PROMPT_TOKENS = 1_000;
const PESSIMISTIC_CHARS_PER_TOKEN = 2.5;
const TRANSCRIPT_CHAR_BUDGET =
  (MODEL_CONTEXT_TOKENS - RESERVED_OUTPUT_TOKENS - RESERVED_PROMPT_TOKENS) *
  PESSIMISTIC_CHARS_PER_TOKEN;

// Output caps, so the model is never handed "whatever is left of the window".
const MINUTES_MAX_OUTPUT_TOKENS = RESERVED_OUTPUT_TOKENS;
const CHAPTER_SUMMARY_MAX_OUTPUT_TOKENS = 600;

// The generation-relevant subset of a Skabelon.
export interface SkabelonSpec {
  prompt: string;
  includeDeltagere: boolean;
  includeBeslutningspunkter: boolean;
  includeDagsorden: boolean;
  includeDato: boolean;
}

// ─── Prompt building ──────────────────────────────────────────────────────────

export function buildSkabelonInstruction(
  spec: SkabelonSpec,
  participants?: string[],
  customPrompt?: string,
): string {
  const parts: string[] = [];
  if (spec.prompt.trim()) parts.push(spec.prompt.trim());

  const categories: string[] = [];
  // The "Dato" tag no longer injects the date into the body — the date lives in
  // the editable document header (see MinutesContent.header) so it isn't rendered
  // twice. `spec.includeDato` is consumed at save time to populate that header.
  if (spec.includeDagsorden) {
    categories.push('- En **Dagsorden**-sektion med mødets punkter.');
  }
  if (spec.includeDeltagere) {
    const names =
      participants && participants.length > 0 ? ` Deltagere: ${participants.join(', ')}.` : '';
    categories.push(`- En **Deltagere**-sektion med mødets deltagere.${names}`);
  }
  if (spec.includeBeslutningspunkter) {
    categories.push('- En **Beslutningspunkter**-sektion der opsummerer de trufne beslutninger.');
  }
  if (categories.length > 0) {
    parts.push(
      'Strukturér referatet med følgende afsnit, og medtag ikke yderligere faste afsnit (fx beslutninger eller resumé) medmindre instruktionen nedenfor beder om det:\n' +
        categories.join('\n'),
    );
  } else if (participants && participants.length > 0) {
    parts.push(`Deltagere i mødet: ${participants.join(', ')}.`);
  }

  if (customPrompt && customPrompt.trim()) {
    parts.push('Følg denne instruktion nøje: ' + customPrompt.trim());
  }
  return parts.join('\n\n');
}

/**
 * Collapse runs of consecutive segments from the same speaker into one turn.
 *
 * hviske emits short utterances, so one person speaking for a minute arrives as many
 * segments, each repeating the `[Taler N] (mm:ss): ` prefix. On a measured 57-minute
 * meeting that was 982 segments versus 126 actual turns — 37% of the prompt's characters
 * and roughly half its tokens spent on labels rather than speech.
 *
 * Returns new objects; the caller's segments are not mutated.
 */
export function mergeConsecutiveSpeakerTurns(segments: TranscriptSegment[]): TranscriptSegment[] {
  const turns: TranscriptSegment[] = [];
  for (const segment of segments) {
    const previous = turns[turns.length - 1];
    if (previous && previous.speaker === segment.speaker) {
      previous.text = `${previous.text} ${segment.text}`.trim();
      previous.end = segment.end;
    } else {
      turns.push({ ...segment });
    }
  }
  return turns;
}

// ─── Generation ───────────────────────────────────────────────────────────────

async function _generateBody(transcriptText: string, instruction: string): Promise<string> {
  const response = await getClient().chat.completions.create({
    model: LLM_MODEL,
    max_tokens: MINUTES_MAX_OUTPUT_TOKENS,
    messages: [
      { role: 'system', content: MINUTES_SYSTEM_PROMPT },
      {
        role: 'user',
        content: `Udarbejd et mødereferat baseret på denne transskription.
${instruction ? `\n${instruction}\n` : ''}
Følg instruktionerne ovenfor nøje — herunder ønsket længde og hvilke afsnit der skal med. Skriv referatet som ét sammenhængende dokument i markdown. Brug overskrifter (##) til afsnit og punktlister hvor det er relevant. Returner KUN selve referatet — ingen forklaringer, ingen JSON og ingen code blocks.

Transskription:
${transcriptText}`,
      },
    ],
  });

  const raw = response.choices[0]?.message?.content ?? '';
  // Strip an accidental markdown code fence if the model wraps the document.
  return raw
    .replace(/^```(?:markdown|md)?\s*/i, '')
    .replace(/```\s*$/, '')
    .trim();
}

async function _summarizeChapter(
  chapterSegments: TranscriptSegment[],
  chapterTitle: string,
): Promise<string> {
  const transcriptText = mergeConsecutiveSpeakerTurns(chapterSegments)
    .map((s) => `[${s.speaker}]: ${s.text}`)
    .join('\n');

  const response = await getClient().chat.completions.create({
    model: LLM_MODEL,
    max_tokens: CHAPTER_SUMMARY_MAX_OUTPUT_TOKENS,
    messages: [
      {
        role: 'user',
        content: `Opsummer mødekapitlet "${chapterTitle}" i korte punkter på dansk (max 8 punkter). Fokus på beslutninger, aftaler og vigtige diskussionspunkter.

${transcriptText}

Returner kun en punktliste.`,
      },
    ],
  });

  return response.choices[0]?.message?.content?.trim() ?? '';
}

/**
 * Generate a referat as a single markdown document, driven by a Skabelon.
 *
 * For long, chaptered transcripts the chapters are summarised first and the
 * referat is written from those summaries, keeping the request within budget.
 */
export async function generateReferatBody(
  transcript: TranscriptSegment[],
  spec: SkabelonSpec,
  participants?: string[],
  chapters?: TranscriptChapter[],
  customPrompt?: string,
): Promise<{ body: string }> {
  const transcriptText = mergeConsecutiveSpeakerTurns(transcript)
    .map((s) => `[${s.speaker}] (${formatTime(s.start)}): ${s.text}`)
    .join('\n');
  const instruction = buildSkabelonInstruction(spec, participants, customPrompt);

  if (chapters && chapters.length > 1 && transcriptText.length > TRANSCRIPT_CHAR_BUDGET) {
    const summaries = await Promise.all(
      chapters.map((ch) => {
        const chapterSegments = ch.segmentIndices.map((i) => transcript[i]).filter(Boolean);
        return _summarizeChapter(chapterSegments, ch.title);
      }),
    );
    const condensed = chapters.map((ch, i) => `## ${ch.title}\n${summaries[i]}`).join('\n\n');
    const body = await _generateBody(condensed, instruction);
    return { body };
  }

  const body = await _generateBody(transcriptText, instruction);
  return { body };
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
