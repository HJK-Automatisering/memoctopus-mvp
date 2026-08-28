import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockComplete = vi.hoisted(() => vi.fn());

vi.mock('openai', () => ({
  default: class {
    chat = { completions: { create: mockComplete } };
  },
}));

import {
  generateReferatBody,
  buildSkabelonInstruction,
  mergeConsecutiveSpeakerTurns,
  SkabelonSpec,
} from './minutes';
import type { TranscriptSegment } from '@/types';

const sampleSegments: TranscriptSegment[] = [
  { speaker: 'Taler 1', start: 0, end: 5, text: 'Vi åbner mødet.' },
  { speaker: 'Taler 2', start: 6, end: 10, text: 'Første punkt på dagsordenen.' },
];

const baseSpec: SkabelonSpec = {
  prompt: 'Lav et kortfattet referat.',
  includeDeltagere: false,
  includeBeslutningspunkter: false,
  includeDagsorden: false,
  includeDato: false,
};

function openaiResponse(content: string) {
  return { choices: [{ message: { content } }] };
}

// ─── buildSkabelonInstruction ─────────────────────────────────────────────────

describe('buildSkabelonInstruction', () => {
  it('includes the base prompt', () => {
    const out = buildSkabelonInstruction(baseSpec);
    expect(out).toContain('Lav et kortfattet referat.');
  });

  it('injects the three categories when toggled on', () => {
    const out = buildSkabelonInstruction(
      { ...baseSpec, includeDeltagere: true, includeBeslutningspunkter: true, includeDagsorden: true },
      ['Anna', 'Bjørn'],
    );
    expect(out).toContain('Deltagere');
    expect(out).toContain('Beslutningspunkter');
    expect(out).toContain('Dagsorden');
    expect(out).toContain('Anna, Bjørn');
  });

  it('lists participants even without the Deltagere category', () => {
    const out = buildSkabelonInstruction(baseSpec, ['Anna']);
    expect(out).toContain('Anna');
  });

  it('appends the custom prompt', () => {
    const out = buildSkabelonInstruction(baseSpec, [], 'Fokus på handlinger');
    expect(out).toContain('Fokus på handlinger');
  });

  it('omits categories that are toggled off', () => {
    const out = buildSkabelonInstruction({ ...baseSpec, includeDagsorden: true });
    expect(out).toContain('Dagsorden');
    expect(out).not.toContain('Beslutningspunkter');
  });

  it('does not inject the date into the body — the Dato tag drives the document header instead', () => {
    const out = buildSkabelonInstruction({ ...baseSpec, includeDato: true });
    expect(out).not.toContain('Dato');
    expect(out).not.toContain('dato');
    // Other categories are unaffected.
    const withAgenda = buildSkabelonInstruction({ ...baseSpec, includeDato: true, includeDagsorden: true });
    expect(withAgenda).toContain('Dagsorden');
  });
});

// ─── generateReferatBody ──────────────────────────────────────────────────────

describe('generateReferatBody', () => {
  beforeEach(() => mockComplete.mockReset());

  it('returns the markdown body from the OpenAI response', async () => {
    mockComplete.mockResolvedValueOnce(openaiResponse('## Referat\n\nMødet blev åbnet.'));

    const result = await generateReferatBody(sampleSegments, baseSpec);

    expect(result.body).toBe('## Referat\n\nMødet blev åbnet.');
  });

  it('strips an accidental markdown code fence', async () => {
    mockComplete.mockResolvedValueOnce(openaiResponse('```markdown\n## Referat\n\nIndhold.\n```'));

    const result = await generateReferatBody(sampleSegments, baseSpec);

    expect(result.body).toBe('## Referat\n\nIndhold.');
  });

  it('includes the transcript text in the prompt', async () => {
    mockComplete.mockResolvedValueOnce(openaiResponse('referat'));

    await generateReferatBody(sampleSegments, baseSpec);

    const userContent = mockComplete.mock.calls[0][0].messages[1].content as string;
    expect(userContent).toContain('Vi åbner mødet.');
  });

  it('includes formatted timestamps in the prompt', async () => {
    mockComplete.mockResolvedValueOnce(openaiResponse('referat'));

    await generateReferatBody([{ speaker: 'Taler 1', start: 65, end: 70, text: 'Hej' }], baseSpec);

    const userContent = mockComplete.mock.calls[0][0].messages[1].content as string;
    expect(userContent).toContain('1:05'); // 65 seconds
  });

  it('feeds the built instruction into the prompt', async () => {
    mockComplete.mockResolvedValueOnce(openaiResponse('referat'));

    await generateReferatBody(sampleSegments, { ...baseSpec, includeDagsorden: true });

    const userContent = mockComplete.mock.calls[0][0].messages[1].content as string;
    expect(userContent).toContain('Dagsorden');
  });

  it('uses configured LLM model', async () => {
    mockComplete.mockResolvedValueOnce(openaiResponse('referat'));

    await generateReferatBody(sampleSegments, baseSpec);

    expect(mockComplete.mock.calls[0][0].model).toBe(process.env.LLM_MODEL ?? 'Qwen/Qwen3.6-27B');
  });
});

// ─── mergeConsecutiveSpeakerTurns ─────────────────────────────────────────────

describe('mergeConsecutiveSpeakerTurns', () => {
  it('joins a run of same-speaker segments into one turn and leaves the input untouched', () => {
    const segments: TranscriptSegment[] = [
      { speaker: 'Taler 1', start: 0, end: 4, text: 'Vi åbner mødet.' },
      { speaker: 'Taler 1', start: 4, end: 9, text: 'Første punkt.' },
      { speaker: 'Taler 2', start: 9, end: 12, text: 'Enig.' },
    ];

    const turns = mergeConsecutiveSpeakerTurns(segments);

    expect(turns).toEqual([
      { speaker: 'Taler 1', start: 0, end: 9, text: 'Vi åbner mødet. Første punkt.' },
      { speaker: 'Taler 2', start: 9, end: 12, text: 'Enig.' },
    ]);
    // The caller's segments must survive — they are rendered elsewhere in the UI.
    expect(segments).toHaveLength(3);
    expect(segments[0].text).toBe('Vi åbner mødet.');
  });
});

// ─── transcript budget trigger ────────────────────────────────────────────────

describe('generateReferatBody chapter-summarisation trigger', () => {
  beforeEach(() => mockComplete.mockReset());

  // Alternating speakers so merging cannot collapse the transcript below the budget.
  const longTranscript: TranscriptSegment[] = Array.from({ length: 200 }, (_, i) => ({
    speaker: `Taler ${(i % 2) + 1}`,
    start: i * 10,
    end: i * 10 + 10,
    text: 'Vi diskuterede budgettet for det kommende år i detaljer. '.repeat(4),
  }));

  const twoChapters = [
    { id: 'ch-0', title: 'Budget', summary: '', startTime: 0, endTime: 1000,
      segmentIndices: Array.from({ length: 100 }, (_, i) => i) },
    { id: 'ch-1', title: 'Personale', summary: '', startTime: 1000, endTime: 2000,
      segmentIndices: Array.from({ length: 100 }, (_, i) => i + 100) },
  ];

  it('summarises per chapter when the merged transcript exceeds the char budget', async () => {
    mockComplete.mockResolvedValue(openaiResponse('- punkt'));

    await generateReferatBody(longTranscript, baseSpec, undefined, twoChapters);

    // Two chapter summaries plus the final referat.
    expect(mockComplete).toHaveBeenCalledTimes(3);
  });

  it('stays single-shot when the transcript is within the char budget', async () => {
    mockComplete.mockResolvedValue(openaiResponse('referat'));

    await generateReferatBody(sampleSegments, baseSpec, undefined, twoChapters);

    expect(mockComplete).toHaveBeenCalledTimes(1);
  });
});
