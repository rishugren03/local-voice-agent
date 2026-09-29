/**
 * U2 — Talk.
 *
 * A microphone, a waveform, and the stages of the turn as they happen. The
 * design goal is that a slow call explains itself: when someone wonders why the
 * agent took four seconds, this screen already showed where the time went, so
 * they do not have to go to a log to find out.
 *
 * Two decisions worth naming:
 *
 *  - The client mints its own token via the API rather than embedding a LiveKit
 *    key. The secret never leaves the machine, and rotating it is one file.
 *  - Local echo cancellation and noise suppression are off by default. The
 *    agent uses Silero VAD on the same audio, and browser AEC aggressively
 *    gates an agent's own synthesised voice — which it then mishears as the
 *    user interrupting. Anyone who hits that can turn it on here rather than
 *    discovering it in their speaker settings.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Chip, Problems, Spinner } from "../components";

import { ApiError, api } from "../api";
// Type-only import. The runtime import is deferred to the moment the user
// actually starts a call, because livekit-client is ~700kB of the bundle and six
// of the seven screens never open a microphone. Keeping it out of the initial
// chunk is most of this app's load time.
import type { Room, RemoteTrack, RemoteParticipant } from "livekit-client";

type Stage = "idle" | "connecting" | "ready" | "disconnected" | "error";

interface Turn {
  who: "you" | "agent";
  text: string;
  reply?: string;
  stt?: number;
  llm?: number;
  tts?: number;
  degraded?: boolean;
}

const PRESETS = [
  { label: "Ask the agent to calculate", text: "What is 12 plus 15?" },
  { label: "Trigger a handoff", text: "I need help with a billing problem." },
  { label: "Ask about the weather", text: "What's the weather in Berlin?" },
];

export default function Talk() {
  const [stage, setStage] = useState<Stage>("idle");
  const [room, setRoom] = useState<string | null>(null);
  const [identity, setIdentity] = useState<string | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [muted, setMuted] = useState(false);
  const [echoCancel, setEchoCancel] = useState(false);

  const [transcript, setTranscript] = useState<Turn[]>([]);
  const transcriptEnd = useRef<HTMLDivElement | null>(null);
  const [level, setLevel] = useState(0);
  const [connected, setConnected] = useState(false);

  const roomRef = useRef<Room | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);

  const teardown = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    void roomRef.current?.disconnect();
    roomRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    void audioCtxRef.current?.close();
    audioCtxRef.current = null;
    setConnected(false);
    setLevel(0);
  }, []);

  // Leaving the page mid-call would otherwise keep publishing audio and holding
  // the room, and the next person to open this screen would land in a room that
  // already has a caller in it.
  useEffect(() => teardown, [teardown]);

  async function connect() {
    setProblems([]);
    setStage("connecting");
    try {
      const grant = await api.mintToken();

      // GetUserMedia before creating the room. If the user denies the mic, this
      // throws here with an error we can explain, rather than leaving a room
      // open that will never hear them.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: echoCancel,
          noiseSuppression: echoCancel,
          autoGainControl: echoCancel,
        },
      });
      streamRef.current = stream;

      const { Room, RoomEvent } = await import("livekit-client");
      const rk = new Room({ adaptiveStream: true, dynacast: true });
      roomRef.current = rk;

      // Attach the agent's audio before connecting, so the first syllable is not
      // dropped while the subscription is still negotiating.
      rk.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind === "audio") {
          audioRef.current?.appendChild(track.attach());
        }
      });

      rk.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        if (track.kind === "audio") {
          track.detach().forEach((el) => el.remove());
        }
      });

      rk.on(RoomEvent.Disconnected, () => {
        setConnected(false);
        setStage("disconnected");
      });

      await rk.connect(grant.url, grant.token);
      await rk.localParticipant.setMicrophoneEnabled(true);
      for (const participant of rk.remoteParticipants.values()) {
        attachExisting(participant, audioRef.current);
      }
      rk.on(RoomEvent.ParticipantConnected, (p: RemoteParticipant) => {
        attachExisting(p, audioRef.current);
      });

      setRoom(grant.room);
      setIdentity(grant.identity);
      setConnected(true);
      setStage("ready");
      watchLevel(stream);
    } catch (e) {
      teardown();
      setStage("error");
      if (e instanceof ApiError) setProblems(e.problems);
      else if (e instanceof DOMException)
        setProblems([`Microphone unavailable: ${e.message}. Check the browser's permission for this origin.`]);
      else setProblems([String(e)]);
    }
  }

  function attachExisting(participant: RemoteParticipant, sink: HTMLAudioElement | null) {
    for (const publication of participant.trackPublications.values()) {
      if (publication.kind === "audio" && publication.track) {
        sink?.appendChild(publication.track.attach());
      }
    }
  }

  // A coarse input meter. It is not a level meter anyone can calibrate against;
  // it answers "is the browser actually hearing me", which is the question that
  // matters when a call fails.
  function watchLevel(stream: MediaStream) {
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const data = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteFrequencyData(data);
      const sum = data.reduce((a, b) => a + b, 0);
      setLevel(Math.min(1, sum / (data.length * 90)));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  async function toggleMute() {
    if (!roomRef.current) return;
    await roomRef.current.localParticipant.setMicrophoneEnabled(muted);
    setMuted(!muted);
  }

  function disconnect() {
    teardown();
    setStage("idle");
  }

  // Follow the newest turn. A transcript that does not scroll itself means
  // reading the turn you just made means scrolling to find it, every time.
  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [transcript]);

  // The live transcript is read back out of the trace database while the call
  // runs, not computed here. The worker owns speech-to-text and the model, so
  // the browser has no way to know what was said — but every turn is written to
  // SQLite as it completes, and the calls screen already reads from there.
  //
  // Polling is keyed on our own identity, which the server generated for us, so
  // two browsers in two tabs are two sessions and neither sees the other's
  // turns. That is the same reason the token mints a random identity rather than
  // taking one from the client.
  useEffect(() => {
    if (!connected || !identity) return;
    let cancelled = false;

    const tick = async () => {
      try {
        const list = await api.listSessions(5, 0, identity);
        const mine = list.sessions.find((s) => s.participant_identity === identity);
        if (!mine) return;
        const session = await api.getSession(mine.session_id);
        if (cancelled) return;
        setTranscript(
          session.calls.map((call) => ({
            who: "you" as const,
            text: call.user_text ?? "",
            reply: call.reply ?? "",
            stt: call.latency.stt ?? undefined,
            llm: call.latency.llm ?? undefined,
            tts: call.latency.tts ?? undefined,
            degraded: call.degraded,
          })),
        );
      } catch {
        // A missed poll is not worth interrupting the call for; the next one is
        // two seconds away and the trace is not going anywhere.
      }
    };

    void tick();
    const timer = setInterval(tick, 1500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [connected, identity]);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Talk</h1>
          <p>
            Talk to the agent directly. Everything here is local — the browser joins
            LiveKit on this machine and the model runs on this machine.
          </p>
        </div>
        <div className="row tight">
          {stage === "ready" && (
            <Chip tone={muted ? "warn" : "ok"} dot>
              {muted ? "muted" : "live"}
            </Chip>
          )}
        </div>
      </div>

      <Problems problems={problems} />

      <div className="talk-grid">
        <div className="card">
          <h2>Call</h2>
          <p className="hint">
            Uses the default room from the worker's configuration. Speak normally;
            interrupting works.
          </p>

          <div className="row" style={{ marginBottom: 12 }}>
            {stage !== "ready" ? (
              <button className="btn primary" onClick={connect} disabled={stage === "connecting"}>
                {stage === "connecting" ? "Connecting…" : "Start call"}
              </button>
            ) : (
              <>
                <button className="btn" onClick={toggleMute}>
                  {muted ? "Unmute" : "Mute"}
                </button>
                <button className="btn danger" onClick={disconnect}>
                  End call
                </button>
              </>
            )}
          </div>

          {connected && (
            <div className={level > 0.75 ? "meter hot" : "meter"}>
              <div style={{ width: `${Math.round(level * 100)}%` }} />
            </div>
          )}
          {connected && (
            <div className="meter-label">
              <span>{room}</span>
              <span>{level > 0.02 ? "hearing you" : "silence"}</span>
            </div>
          )}

          <label className="check" style={{ marginTop: 12 }}>
            <input
              type="checkbox"
              checked={echoCancel}
              disabled={connected}
              onChange={(e) => setEchoCancel(e.target.checked)}
            />
            <span>Browser echo cancellation</span>
          </label>
          <p className="field-hint" style={{ marginTop: 2 }}>
            Off by default. Turn it on if you hear your own echo, but expect the agent
            to occasionally cut itself off mid-sentence — it is designed for a
            headset, not speakers.
          </p>

          {room && (
            <div className="notice" style={{ marginTop: 12 }}>
              Joined as <span className="mono">{identity}</span> in{" "}
              <span className="mono">{room}</span>. The full turn-by-turn record lands
              on the Calls screen.
            </div>
          )}
        </div>

        <div className="card">
          <h2>Live transcript</h2>
          <p className="hint">
            What the browser heard and played. Interruption and barge-in events are in
            the call record.
          </p>

          {stage === "idle" && (
            <>
              <div className="empty" style={{ padding: "18px 0 12px" }}>
                Start a call to begin.
              </div>
              <div className="list">
                {PRESETS.map((preset) => (
                  <div key={preset.text} className="faint" style={{ fontSize: 12.5 }}>
                    Try: “{preset.text}”
                  </div>
                ))}
              </div>
            </>
          )}

          {stage === "connecting" && <Spinner />}

          {stage === "ready" && (
            <div className="transcript">
              {transcript.length === 0 && (
                <div className="empty" style={{ padding: 16 }}>
                  Listening. Say something — the meter above moves when it hears you,
                  and the turn appears here once the agent has answered.
                </div>
              )}
              {transcript.map((turn, i) => (
                <div key={i} className="turn user">
                  <div className="who">you</div>
                  {turn.text}
                </div>
              ))}
              {transcript.map((turn, i) => (
                <div key={`reply-${i}`} className={`turn agent ${turn.degraded ? "degraded" : ""}`}>
                  <div className="who">
                    agent
                    {turn.degraded && " · degraded"}
                  </div>
                  {turn.reply || <span className="faint">no reply recorded</span>}
                  {(turn.stt !== undefined || turn.llm !== undefined) && (
                    <div className="stage">
                      <span>stt {turn.stt?.toFixed(2) ?? "?"}s</span>
                      <span>llm {turn.llm?.toFixed(2) ?? "?"}s</span>
                      <span>tts {turn.tts?.toFixed(2) ?? "?"}s</span>
                    </div>
                  )}
                </div>
              ))}
              <div ref={transcriptEnd} />
            </div>
          )}

          {stage === "disconnected" && <div className="empty">Call ended.</div>}
        </div>
      </div>
    </>
  );
}
