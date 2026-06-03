from __future__ import annotations

import logging

import discord
import discord.ext.voice_recv as voice_recv
import discord.ext.voice_recv.router as router
import discord.opus
import discord.voice_state as voice_state


logging.getLogger("discord.ext.voice_recv").setLevel(logging.ERROR)
logging.getLogger("discord.ext.voice_recv.router").setLevel(logging.ERROR)
logging.getLogger("discord.opus").setLevel(logging.ERROR)


voice_recv.reader.PacketDecryptor.supported_modes = [
    "aead_xchacha20_poly1305_rtpsize",
    "xsalsa20_poly1305_lite",
    "xsalsa20_poly1305_suffix",
    "xsalsa20_poly1305",
]


_original_decode = discord.opus.Decoder.decode


def _safe_decode(self, data, *, fec=False):
    try:
        return _original_decode(self, data, fec=fec)
    except discord.opus.OpusError:
        return b"\x00" * 3840


discord.opus.Decoder.decode = _safe_decode


_original_decode_packet = router.PacketDecoder._decode_packet


def _patched_decode_packet(self, packet):
    sink = getattr(self, "sink", None)
    if not sink:
        return _original_decode_packet(self, packet)

    try:
        recv_client = getattr(sink, "voice_client", None)
        connection = getattr(recv_client, "_connection", None) if recv_client else None
        dave_session = getattr(connection, "dave_session", None) if connection else None
        davey = getattr(voice_state, "davey", None)

        if dave_session and davey and getattr(dave_session, "ready", False):
            user_id = recv_client._get_id_from_ssrc(packet.ssrc)
            if user_id:
                decrypted = dave_session.decrypt(
                    media_type=davey.MediaType.audio,
                    user_id=user_id,
                    packet=packet.decrypted_data,
                )
                packet.decrypted_data = decrypted or b""
            else:
                packet.decrypted_data = b""
    except Exception:
        pass

    return _original_decode_packet(self, packet)


router.PacketDecoder._decode_packet = _patched_decode_packet


class PatchedVoiceRecvClient(voice_recv.VoiceRecvClient):
    @property
    def supported_modes(self):
        return tuple(voice_recv.reader.PacketDecryptor.supported_modes)

    def _remove_ssrc(self, user_id: int) -> None:
        reader = self._reader
        if reader and not isinstance(reader, discord.utils._MissingSentinel):
            try:
                super()._remove_ssrc(user_id)
                return
            except Exception:
                pass

        ssrc = self._id_to_ssrc.pop(user_id, None)
        if ssrc:
            self._ssrc_to_id.pop(ssrc, None)
