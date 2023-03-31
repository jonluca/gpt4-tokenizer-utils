# GPT4 Tokenizer

[![Build](https://github.com/jonluca/gpt4-tokenizer-utils/actions/workflows/main.yml/badge.svg)](https://github.com/jonluca/gpt4-tokenizer-utils/actions/workflows/main.yml)
[![NPM Version](https://img.shields.io/npm/v/gpt4-tokenizer-utils.svg)](https://www.npmjs.com/package/gpt4-tokenizer)
[![NPM Downloads](https://img.shields.io/npm/dt/gpt4-tokenizer-utils.svg)](https://www.npmjs.com/package/gpt4-tokenizer)

This is a isomorphic TypeScript tokenizer for OpenAI's GPT-4 model. It also includes some utility functions for tokenizing and encoding text for use with the GPT-4 model.

It will work in all cases that `TextEncoder` and `TextDecoder` are globals.

## Usage

First, install:

```shell
yarn add gpt4-tokenizer
```

### Estimate token length

```typescript
import GPT4Tokenizer from 'gpt4-tokenizer';

const tokenizer = new GPT4Tokenizer({ type: 'gpt3' }); // or 'codex'
const str = 'hello 👋 world 🌍';
const estimatedTokenCount = tokenizer.estimateTokenCount(str); // 7
```

### Chunk by token length

```typescript
import GPT4Tokenizer from 'gpt4-tokenizer';

const tokenizer = new GPT4Tokenizer({ type: 'gpt3' }); // or 'codex'
const str = 'A very long string...';
const estimatedTokenCount = tokenizer.chunkText(str, 5); // 7
```

## Reference

This library is based on the following:

- [gpt3-tokenzier](https://github.com/jonluca/gpt4-tokenizer-utils)
- [OpenAI Tokenizer Page Source](https://beta.openai.com/tokenizer?view=bpe)
- [gpt-3-encoder](https://github.com/latitudegames/GPT-3-Encoder)

The main difference between this library and gpt-3-encoder is that this library supports both `gpt3` and `codex` tokenization (The dictionary is taken directly from OpenAI so the tokenization result is on par with the OpenAI Playground). Also Map API is used instead of JavaScript objects, especially the `bpeRanks` object, which should see some performance improvement.

## License

[MIT](./LICENSE)
