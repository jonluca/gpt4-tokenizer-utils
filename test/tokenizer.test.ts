import GPT4Tokenizer from '../src';

describe('gpt3 tokenizer test', () => {
  it('works with empty string', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'gpt3' });
    const str = '';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with a single space', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'gpt3' });
    const str = ' ';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([220]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with a single tab', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'gpt3' });
    const str = '\t';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([197]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with some simple text', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'gpt3' });
    const str = 'This is some text';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([1212, 318, 617, 2420]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with multi-token word', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'gpt3' });
    const str = 'indivisible';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([521, 452, 12843]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with emojis', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'gpt3' });
    const str = 'hello 👋 world 🌍';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([31373, 50169, 233, 995, 12520, 234, 235]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with code using codex', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'codex' });
    const str = "def main():\n    print('hello world')";
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([4299, 1388, 33529, 198, 50258, 3601, 10786, 31373, 995, 11537]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('works with javascript object property strings', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'codex' });
    const str = 'some_code toString some_more_code';
    const encoded = tokenizer.encode(str);
    expect(encoded).toEqual([11246, 62, 8189, 284, 10100, 617, 62, 3549, 62, 8189]);
    expect(tokenizer.decode(encoded)).toEqual(str);
  });

  it('chunks text appropriately', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'codex' });
    const str = 'A string with some length greater than 5';
    const encoded = tokenizer.chunkText(str, 5);
    expect(encoded.map((c) => c.text)).toEqual(['A string with some length', ' greater than 5']);
  });
  it('estimates token length correctly', () => {
    const tokenizer = new GPT4Tokenizer({ type: 'codex' });
    const str = 'A string with some length greater than 5';
    const encoded = tokenizer.estimateTokenCount(str);
    expect(encoded).toEqual(8);
  });
});
