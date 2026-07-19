import ArrayKeyedMap from 'array-keyed-map';

import bpeVocab from './bpe-vocab';
import bpeRegex from './bpe-regex';
import encodings from './encodings';

const range = (x: number, y: number) => {
  const res = Array.from(Array(y).keys()).slice(x);
  return res;
};

const ord = (x: string): number => {
  return x.charCodeAt(0);
};

const chr = (n: number): string => {
  return String.fromCharCode(n);
};

export class GPT4Tokenizer {
  private vocab: string;
  private nMergedSpaces: number;
  private nVocab: number;

  private encodings: { [key: string]: number };
  private decodings: { [key: number]: string };

  private byteEncoder: Map<number, string>;
  private byteDecoder: Map<string, number>;

  private bpeRanks: ArrayKeyedMap<[string, string], number>;
  private cache: { [key: string]: string };
  private encodeCache = new Map<string, number[]>();

  private textEncoder: TextEncoder;
  private textDecoder: TextDecoder;
  constructor(options: { type: 'gpt3' | 'gpt4' | 'codex' }) {
    this.encodings = encodings;
    this.vocab = bpeVocab;
    this.nMergedSpaces = options.type === 'codex' ? 24 : 0;
    this.nVocab = 50257 + this.nMergedSpaces;
    this.decodings = {};
    this.bpeRanks = new ArrayKeyedMap<[string, string], number>();
    this.byteEncoder = new Map();
    this.byteDecoder = new Map();
    this.cache = {};

    this.textEncoder = new TextEncoder();
    this.textDecoder = new TextDecoder();

    this.initialize();
  }

  private initialize() {
    if (this.vocab.length < 100) {
      throw new Error('Tokenizer vocab file did not load correctly');
    }

    const vocabLines = this.vocab.split('\n');
    const bpeMerges: [string, string][] = vocabLines
      .slice(1, vocabLines.length - 1)
      .map((line: string) => line.split(/(\s+)/).filter((part: string) => part.trim().length > 0)) as [
      string,
      string,
    ][];

    // add merged spaces for codex tokenizer
    if (this.nMergedSpaces > 0) {
      for (let i = 1; i < this.nMergedSpaces; i++) {
        for (let j = 1; j < this.nMergedSpaces; j++) {
          if (i + j <= this.nMergedSpaces) {
            bpeMerges.push(['\u0120'.repeat(i), '\u0120'.repeat(j)]);
          }
        }
      }

      for (let i = 0; i < this.nMergedSpaces; i++) {
        this.encodings['\u0120'.repeat(i + 2)] = this.nVocab - this.nMergedSpaces + i;
      }
    }

    for (const key of Object.keys(this.encodings)) {
      this.decodings[this.encodings[key]] = key;
    }

    this.byteEncoder = this.bytesToUnicode();

    this.byteEncoder.forEach((value, key) => {
      this.byteDecoder.set(value, key);
    });

    this.zip(this.bpeRanks, bpeMerges, range(0, bpeMerges.length));
  }

  private zip<X, Y>(result: Map<X, Y>, x: X[], y: Y[]): Map<X, Y> {
    let length = x.length;
    for (let idx = 0; idx < length; idx++) {
      result.set(x[idx], y[idx]);
    }

    return result;
  }

  private bytesToUnicode(): Map<number, string> {
    const bs = range(ord('!'), ord('~') + 1).concat(
      range(ord('\xa1'), ord('\xac') + 1),
      range(ord('\xae'), ord('\xff') + 1),
    );

    let cs: number[] = bs.slice();
    let n = 0;

    for (let b = 0; b < Math.pow(2, 8); b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(Math.pow(2, 8) + n);
        n = n + 1;
      }
    }

    let csStr = cs as unknown as string[];
    for (let i = 0; i < cs.length; i++) {
      csStr[i] = chr(cs[i]);
    }

    const result = new Map<number, string>();
    this.zip(result, bs, csStr);
    return result;
  }

  getPairs(word: string[]): Set<[string, string]> {
    const pairs = new Set<[string, string]>();
    let prevChar = word[0];

    for (let i = 1; i < word.length; i++) {
      const char = word[i];
      pairs.add([prevChar, char]);
      prevChar = char;
    }

    return pairs;
  }

  bpe(token: string): string {
    if (Object.prototype.hasOwnProperty.call(this.cache, token)) {
      return this.cache[token];
    }

    let word: string[] | string = token.split('');

    let pairs = this.getPairs(word);

    if (pairs.size === 0) {
      return token;
    }

    while (true) {
      let minRank = 1e11;
      let bigram: [string, string] = ['', ''];
      for (const pair of Array.from(pairs)) {
        const rank = this.bpeRanks.get(pair);
        let realRank = isNaN(rank as number) ? 1e11 : (rank as number);
        if (realRank < minRank) {
          bigram = pair;
          minRank = realRank;
        }
      }

      if (!this.bpeRanks.has(bigram)) {
        break;
      }

      const first = bigram[0];
      const second = bigram[1];
      let newWord: string[] = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord = newWord.concat(word.slice(i));
          break;
        }
        newWord = newWord.concat(word.slice(i, j));
        i = j;

        if (word[i] === first && i < word.length - 1 && word[i + 1] === second) {
          newWord.push(first + second);
          i = i + 2;
        } else {
          newWord.push(word[i]);
          i = i + 1;
        }
      }

      word = newWord;
      if (word.length === 1) {
        break;
      } else {
        pairs = this.getPairs(word);
      }
    }

    word = word.join(' ');
    this.cache[token] = word;

    return word;
  }

  encode(text: string): number[] {
    const bpeTokens: number[] = [];
    const matches = text.match(bpeRegex) || [];

    for (let token of matches) {
      const cacheKey = token;
      let newTokens = this.encodeCache.get(cacheKey);
      if (!newTokens) {
        token = Array.from(this.encodeUtf8(token))
          .map((x) => this.byteEncoder.get(x))
          .join('');
        newTokens = this.bpe(token)
          .split(' ')
          .map((x) => this.encodings[x]);
        this.encodeCache.set(cacheKey, newTokens);
      }
      for (let i = 0; i < newTokens.length; i++) {
        bpeTokens.push(newTokens[i]);
      }
    }

    return bpeTokens;
  }

  encodeUtf8(text: string): Uint8Array {
    return this.textEncoder.encode(text);
  }

  decodeUtf8(bytes: Uint8Array): string {
    return this.textDecoder.decode(bytes);
  }
  decode(tokens: number[]): string {
    return this.decodeUtf8(this.decodeTokensToBytes(tokens));
  }

  estimateTokenCount(input: string): number {
    let count: number = 0;
    const matches = input.match(bpeRegex) || [];

    for (let token of matches) {
      token = Array.from(this.encodeUtf8(token))
        .map((x) => this.byteEncoder.get(x))
        .join('');
      const newTokens = this.bpe(token).split(' ');
      count += newTokens.length;
    }

    return count;
  }

  chunkText(text: string, maxTokensPerChunk: number): { text: string; bpe: number[] }[] {
    if (!Number.isInteger(maxTokensPerChunk) || maxTokensPerChunk <= 0) {
      throw new RangeError('maxTokensPerChunk must be a positive integer');
    }

    const encoded = this.encode(text);
    const safeBoundaries = this.getUtf8SafeTokenBoundaries(encoded);
    const chunks: { text: string; bpe: number[] }[] = [];
    let start = 0;

    while (start < encoded.length) {
      let end = Math.min(start + maxTokensPerChunk, encoded.length);

      while (end > start && !safeBoundaries.has(end)) {
        end--;
      }

      if (end === start) {
        throw new RangeError('maxTokensPerChunk is too small to keep a Unicode character intact');
      }

      const chunk = encoded.slice(start, end);
      chunks.push({
        text: this.decode(chunk),
        bpe: chunk,
      });
      start = end;
    }

    return chunks;
  }

  private decodeTokensToBytes(tokens: number[]): Uint8Array {
    const bytes: number[] = [];

    for (const token of tokens) {
      for (const character of this.decodings[token]) {
        bytes.push(this.byteDecoder.get(character) as number);
      }
    }

    return new Uint8Array(bytes);
  }

  private getUtf8SafeTokenBoundaries(tokens: number[]): Set<number> {
    const boundaries = new Set<number>([0]);
    let continuationBytes = 0;

    for (let tokenIndex = 0; tokenIndex < tokens.length; tokenIndex++) {
      for (const byte of this.decodeTokensToBytes([tokens[tokenIndex]])) {
        if (continuationBytes > 0) {
          if ((byte & 0xc0) !== 0x80) {
            throw new Error('Encoded token stream is not valid UTF-8');
          }
          continuationBytes--;
        } else if (byte <= 0x7f) {
          continue;
        } else if (byte >= 0xc2 && byte <= 0xdf) {
          continuationBytes = 1;
        } else if (byte >= 0xe0 && byte <= 0xef) {
          continuationBytes = 2;
        } else if (byte >= 0xf0 && byte <= 0xf4) {
          continuationBytes = 3;
        } else {
          throw new Error('Encoded token stream is not valid UTF-8');
        }
      }

      if (continuationBytes === 0) {
        boundaries.add(tokenIndex + 1);
      }
    }

    if (continuationBytes !== 0) {
      throw new Error('Encoded token stream ended inside a UTF-8 character');
    }

    return boundaries;
  }
}
export default GPT4Tokenizer;
