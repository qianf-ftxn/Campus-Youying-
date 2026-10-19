"""Incremental delivery with the existing output rules applied before release."""
from dataclasses import dataclass
import asyncio
import re
from contextlib import suppress

INTERNAL_SAFETY_LABELS = ("ACTIVE", "MONITORING", "effectiveRisk", "carryoverRisk", "currentRisk", "风险等级", "FALSE_POSITIVE")
PHONE_PATTERN = r"(?<!\d)(?:\d[ -]?){7,13}(?!\d)"
SAFETY_OPENING = "我很在意你现在的安全。请先远离可能伤害自己的物品，联系身边可信任的人陪伴；如果危险正在发生，请立即联系当地紧急服务。你现在是否安全？\n\n"
MONITORING_OPENING = "先关心一下，你最近还好吗，现在是否安全？\n\n"


def verified_contact_numbers(knowledge):
    evidence = " ".join(item.content for item in knowledge if item.source_url and not item.is_demo)
    return {re.sub(r"\D", "", phone) for phone in re.findall(PHONE_PATTERN, evidence)}


class StreamCancelled(BaseException):
    """Transport closed: do not turn cancellation into another generated reply."""


@dataclass
class StreamedReply:
    text: str
    error: str = ""


async def collect_guarded_stream(client, messages, send, fallback: str, is_cancelled=None,
                                verified_phones=None) -> StreamedReply:
    pending, delivered, error = "", [], ""
    # 保留短尾部，防止标签跨模型 token 边界时前半段已经发出。
    hold = max(map(len, INTERNAL_SAFETY_LABELS)) - 1
    def release(text):
        if verified_phones is not None:
            text = re.sub(PHONE_PATTERN, lambda match: match[0] if re.sub(r"\D", "", match[0]) in verified_phones
                          else "[未核实号码已省略]", text)
        if text:
            send(text)
            delivered.append(text)

    async def tokens():
        source = client.stream(messages).__aiter__()
        pending = None
        try:
            while True:
                pending = asyncio.create_task(source.__anext__())
                while not pending.done():
                    if is_cancelled and is_cancelled():
                        raise StreamCancelled()
                    await asyncio.wait({pending}, timeout=.1)
                try:
                    yield pending.result()
                except StopAsyncIteration:
                    break
        finally:
            if pending and not pending.done():
                pending.cancel()
                with suppress(asyncio.CancelledError):
                    await pending
            await source.aclose()
    try:
        async for delta in tokens():
            pending += delta
            positions = [pending.find(label) for label in INTERNAL_SAFETY_LABELS if label in pending]
            if positions:
                pending = pending[:min(positions)] + "\n\n这部分回答暂时无法继续提供，请换个问法后重试。"
                error = "stream blocked by existing internal-label rule"
                break
            end = max(0, len(pending) - hold)
            if verified_phones is not None:
                # 整个数字串留到结束分隔符再检查，不能先泄露号码前半段。
                for number in re.finditer(r"\d[\d -]*", pending):
                    if number.start() < end and (number.end() > end or number.end() == len(pending)):
                        end = number.start()
                        break
            if end:
                release(pending[:end])
                pending = pending[end:]
        if not ("".join(delivered) + pending).strip():
            raise RuntimeError("Model returned empty streamed reply")
    except Exception as exc:
        error = type(exc).__name__
        pending += "\n\n回答暂时中断，请稍后重试。" if delivered or pending else fallback
    release(pending)
    return StreamedReply("".join(delivered), error)
