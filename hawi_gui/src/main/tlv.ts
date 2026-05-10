/**
 * TLV (Type-Length-Value) framing — TypeScript counterpart of hawi_engine.tlv.
 * Wire layout: [type:1B][length:4B BE][value:length B]
 *   type 0x01 = JSON_FRAME (UTF-8 JSON object)
 *   type 0x02 = BINARY_BLOB (reserved, Plan 5)
 *   others   = reserved
 */

export const TYPE_JSON_FRAME = 0x01;
export const TYPE_BINARY_BLOB = 0x02;
export const DEFAULT_MAX_FRAME_SIZE = 16 * 1024 * 1024;
const HEADER_LEN = 5;

export function encodeFrame(typeByte: number, value: Uint8Array): Buffer {
  if (typeByte < 0 || typeByte > 0xff) {
    throw new RangeError(`type byte out of range: ${typeByte}`);
  }
  const header = Buffer.alloc(HEADER_LEN);
  header.writeUInt8(typeByte, 0);
  header.writeUInt32BE(value.length, 1);
  return Buffer.concat([header, value]);
}

export function encodeJsonFrame(obj: unknown): Buffer {
  const body = Buffer.from(JSON.stringify(obj), "utf-8");
  return encodeFrame(TYPE_JSON_FRAME, body);
}

/**
 * Buffered TLV decoder. Feed it Buffer chunks; iterate the yielded
 * [typeByte, value] pairs as soon as full frames arrive.
 */
export class TLVDecoder {
  private buf: Buffer = Buffer.alloc(0);

  constructor(private readonly maxFrameSize = DEFAULT_MAX_FRAME_SIZE) {}

  push(chunk: Buffer): Array<{ typeByte: number; value: Buffer }> {
    this.buf = this.buf.length === 0 ? chunk : Buffer.concat([this.buf, chunk]);
    const out: Array<{ typeByte: number; value: Buffer }> = [];

    while (this.buf.length >= HEADER_LEN) {
      const typeByte = this.buf.readUInt8(0);
      const length = this.buf.readUInt32BE(1);
      if (length > this.maxFrameSize) {
        throw new RangeError(`frame length ${length} exceeds max frame size ${this.maxFrameSize}`);
      }
      const total = HEADER_LEN + length;
      if (this.buf.length < total) break;

      const value = this.buf.subarray(HEADER_LEN, total);
      out.push({ typeByte, value: Buffer.from(value) });
      this.buf = this.buf.subarray(total);
    }

    return out;
  }
}
