import { describe, expect, it } from "vitest";
import { TLVDecoder, TYPE_JSON_FRAME, encodeFrame, encodeJsonFrame } from "./tlv";

describe("tlv", () => {
  it("encodes a JSON frame with the canonical layout", () => {
    const framed = encodeJsonFrame({ hello: 1 });
    const body = Buffer.from(JSON.stringify({ hello: 1 }), "utf-8");
    expect(framed[0]).toBe(0x01);
    expect(framed.readUInt32BE(1)).toBe(body.length);
    expect(framed.subarray(5)).toEqual(body);
  });

  it("rejects an out-of-range type byte", () => {
    expect(() => encodeFrame(-1, Buffer.alloc(0))).toThrow(RangeError);
    expect(() => encodeFrame(256, Buffer.alloc(0))).toThrow(RangeError);
  });

  it("decodes a single frame", () => {
    const decoder = new TLVDecoder();
    const out = decoder.push(encodeJsonFrame({ a: 1 }));
    expect(out).toHaveLength(1);
    expect(out[0].typeByte).toBe(TYPE_JSON_FRAME);
    expect(JSON.parse(out[0].value.toString("utf-8"))).toEqual({ a: 1 });
  });

  it("decodes back-to-back frames", () => {
    const decoder = new TLVDecoder();
    const buf = Buffer.concat([encodeJsonFrame({ a: 1 }), encodeJsonFrame({ b: 2 })]);
    const out = decoder.push(buf);
    expect(out).toHaveLength(2);
    expect(JSON.parse(out[0].value.toString("utf-8"))).toEqual({ a: 1 });
    expect(JSON.parse(out[1].value.toString("utf-8"))).toEqual({ b: 2 });
  });

  it("buffers a frame split across two pushes", () => {
    const decoder = new TLVDecoder();
    const framed = encodeJsonFrame({ split: true });
    const first = decoder.push(framed.subarray(0, 3));
    expect(first).toHaveLength(0);
    const second = decoder.push(framed.subarray(3));
    expect(second).toHaveLength(1);
    expect(JSON.parse(second[0].value.toString("utf-8"))).toEqual({ split: true });
  });

  it("yields unknown type bytes for the caller to filter", () => {
    const decoder = new TLVDecoder();
    const known = encodeJsonFrame({ keep: 1 });
    const unknown = encodeFrame(0x42, Buffer.from("opaque", "utf-8"));
    const out = decoder.push(Buffer.concat([known, unknown]));
    expect(out).toHaveLength(2);
    expect(out[0].typeByte).toBe(TYPE_JSON_FRAME);
    expect(out[1].typeByte).toBe(0x42);
  });
});
