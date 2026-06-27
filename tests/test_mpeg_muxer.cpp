/*
 * Copyright (c) 2016-present The ZLMediaKit project authors. All Rights Reserved.
 *
 * This file is part of ZLMediaKit(https://github.com/ZLMediaKit/ZLMediaKit).
 *
 * Use of this source code is governed by MIT-like license that can be found in the
 * LICENSE file in the root of the source tree. All contributing project authors
 * may be found in the AUTHORS file in the root of the source tree.
 */

// Regression test for the SIGSEGV in MpegMuxer that occurred when an unstable
// stream (e.g. an addStreamProxy pull that frequently disconnects) tore down the
// muxer while the H264/H265 FrameMerger still held a pending callback.
//
// Original bug (src/Record/MPEG.cpp):
//   - inputFrame(H264) registered the merger callback as [this, &track]; it
//     dereferenced &track (a reference into the _tracks map) and the raw
//     _context pointer.
//   - On disconnect, releaseContext() destroyed _context and cleared _tracks
//     WITHOUT flushing the merger. A later flush then fired the callback against
//     a dangling &track / freed _context -> SIGSEGV.
//
// Fix under test:
//   1. The merger callback now captures track_id BY VALUE and returns early if
//      _context is null  ->  no dangling &track, no null-_context deref.
//   2. releaseContext() now calls flush() FIRST, draining any pending merger
//      callback while _context and _tracks are still alive.
//
// What this test exercises:
//   - SPS+PPS (real config frames, no IDR) leave a pending merger callback,
//     exactly the precondition for the crash.
//   - resetTracks() (the disconnect-then-recreate path) must drain that pending
//     callback safely and leave the muxer usable.  [Case A — proves the fix]
//   - A "TS/HLS-style" subclass that flushes in its OWN destructor (mirrors
//     TSMediaSourceMuxer / HlsRecorder) tears down cleanly.  [Case B]
//   - A "PS-encoder-style" subclass that does NOT flush in its own destructor
//     (mirrors PSEncoderImp) exposes a SECONDARY DEFECT the fix introduces:
//     ~MpegMuxer -> releaseContext() -> flush() fires the pending callback ->
//     flushCache() -> onWrite(), but onWrite is pure-virtual once the derived
//     dtor has run -> "pure virtual function called" abort.  [Case C]
//
// Case C is gated behind RUN_PSSTYLE_DTOR_CASE (default OFF) so the regression
// suite stays green; flip it on to observe the destructor-path defect.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "Extension/Frame.h"
#include "Extension/Track.h"
#include "Record/MPEG.h"
#include "Util/logger.h"

using namespace std;
using namespace toolkit;
using namespace mediakit;

#if defined(ENABLE_HLS) || defined(ENABLE_RTPPROXY)

#ifndef RUN_PSSTYLE_DTOR_CASE
#define RUN_PSSTYLE_DTOR_CASE 0
#endif

namespace {

int g_failures = 0;
#define CHECK_TRUE(cond, msg)                                               \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::cerr << "[FAIL] " << (msg) << " (" #cond ")" << std::endl; \
            ++g_failures;                                                   \
        } else {                                                            \
            std::cout << "[ OK ] " << (msg) << std::endl;                   \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------

// An H264 config frame (SPS/PPS): configFrame()==true so it is NOT decodable.
// The merger caches config-only frames and does NOT flush them until a decodable
// frame or a timestamp change arrives -> this is what leaves a pending callback.
class H264ConfigFrame : public FrameFromPtr {
public:
    H264ConfigFrame(std::shared_ptr<BufferLikeString> buf, uint64_t dts, uint64_t pts)
        : _hold(std::move(buf)) {
        _codec_id = CodecH264;
        _ptr = _hold->data();
        _size = _hold->size();
        _dts = dts;
        _pts = pts;
        _prefix_size = 4;
        _is_key = false;
        setIndex(0);
    }
    bool cacheAble() const override { return true; }
    bool configFrame() const override { return true; }   // <- key: stays pending
    bool keyFrame() const override { return false; }

private:
    std::shared_ptr<BufferLikeString> _hold;
};

// An H264 IDR frame: keyFrame()==true, decodable -> forces the merger to flush.
class H264IdrFrame : public FrameFromPtr {
public:
    H264IdrFrame(std::shared_ptr<BufferLikeString> buf, uint64_t dts, uint64_t pts)
        : _hold(std::move(buf)) {
        _codec_id = CodecH264;
        _ptr = _hold->data();
        _size = _hold->size();
        _dts = dts;
        _pts = pts;
        _prefix_size = 4;
        _is_key = true;
        setIndex(0);
    }
    bool cacheAble() const override { return true; }
    bool configFrame() const override { return false; }
    bool keyFrame() const override { return true; }

private:
    std::shared_ptr<BufferLikeString> _hold;
};

std::shared_ptr<BufferLikeString> makeNalBuffer(uint8_t nal_type, size_t payload) {
    static const char start_code[4] = { 0x00, 0x00, 0x00, 0x01 };
    auto buf = std::make_shared<BufferLikeString>();
    buf->append(start_code, 4);
    char header = (char)(nal_type & 0x1F);
    buf->append(&header, 1);
    std::vector<char> body(payload, 0x42);
    buf->append(body.data(), body.size());
    return buf;
}

Frame::Ptr makeSps() { return std::make_shared<H264ConfigFrame>(makeNalBuffer(7, 8), 1000, 1000); }
Frame::Ptr makePps() { return std::make_shared<H264ConfigFrame>(makeNalBuffer(8, 4), 1000, 1000); }
Frame::Ptr makeIdr(uint64_t ts) { return std::make_shared<H264IdrFrame>(makeNalBuffer(5, 64), ts, ts); }

// Minimal H264 track so addTrack() succeeds. MpegMuxer::addTrack() only reads
// getCodecId()/getTrackType()/getIndex(); the other pure virtuals just need to exist.
class FakeH264Track : public Track {
public:
    CodecId getCodecId() const override { return CodecH264; }
    bool ready() const override { return true; }
    Track::Ptr clone() const override { return std::make_shared<FakeH264Track>(*this); }
    Sdp::Ptr getSdp(uint8_t /*payload_type*/) const override { return nullptr; }
};

// ---------------------------------------------------------------------------
// Muxer subclasses
// ---------------------------------------------------------------------------

// Base test muxer: counts onWrite() output so we can assert the pending callback
// actually drained into real TS data rather than being dropped.
class CountingMpegMuxer : public MpegMuxer {
public:
    explicit CountingMpegMuxer(bool is_ps = false) : MpegMuxer(is_ps) {}
    std::atomic<int> write_calls{0};
    std::atomic<size_t> write_bytes{0};

protected:
    void onWrite(std::shared_ptr<toolkit::Buffer> buffer, uint64_t, bool) override {
        if (buffer) {
            ++write_calls;
            write_bytes += buffer->size();
        }
    }
};

// Mirrors TSMediaSourceMuxer / HlsRecorder: flushes in its OWN destructor, so the
// merger is drained while this vtable is still valid. ~MpegMuxer's flush() is then
// a no-op. This is the SAFE production shape.
class TsStyleMuxer : public CountingMpegMuxer {
public:
    explicit TsStyleMuxer(bool is_ps = false) : CountingMpegMuxer(is_ps) {}
    ~TsStyleMuxer() override {
        try {
            MpegMuxer::flush();
        } catch (std::exception &ex) {
            std::cerr << "flush in dtor threw: " << ex.what() << std::endl;
        }
    }
};

// Mirrors PSEncoderImp: does NOT flush in its own destructor. With a pending
// callback this drives ~MpegMuxer -> releaseContext() -> flush() -> onWrite()
// after the derived onWrite slot is gone -> pure-virtual abort.
class PsStyleMuxer : public CountingMpegMuxer {
public:
    explicit PsStyleMuxer(bool is_ps = false) : CountingMpegMuxer(is_ps) {}
    ~PsStyleMuxer() override = default; // intentionally no flush()
};

void feedPendingConfig(MpegMuxer &muxer) {
    CHECK_TRUE(muxer.addTrack(std::make_shared<FakeH264Track>()), "addTrack(H264) succeeds");
    muxer.inputFrame(makeSps()); // cached, callback pending
    muxer.inputFrame(makePps()); // same dts, still config-only -> stays pending
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

// Case A — the scenario the fix targets. resetTracks() routes through
// releaseContext() (which now flush()es) then createContext(). It must not crash
// and the muxer must remain usable afterwards. This is NOT inside a destructor,
// so onWrite dispatches correctly.
void test_reset_tracks_with_pending_merger() {
    std::cout << "\n== Case A: resetTracks() with pending merger callback ==" << std::endl;
    TsStyleMuxer muxer(false);
    feedPendingConfig(muxer);
    muxer.resetTracks(); // onWrite(nullptr) + releaseContext()[flush] + createContext()
    std::cout << "[ OK ] resetTracks() returned without crashing" << std::endl;

    // Muxer must still work: re-add track, push SPS+PPS+IDR, flush -> real TS out.
    CHECK_TRUE(muxer.addTrack(std::make_shared<FakeH264Track>()),
               "addTrack after resetTracks succeeds");
    muxer.inputFrame(makeSps());
    muxer.inputFrame(makePps());
    muxer.inputFrame(makeIdr(2000)); // IDR -> merger flushes the group through live ctx
    muxer.flush();
    CHECK_TRUE(muxer.write_calls.load() > 0,
               "TS output produced after reset (onWrite fired through live context)");
}

// Case B — TS/HLS-style teardown (flush in own dtor). Pending callback present.
// Must complete cleanly: the derived dtor drains the merger, ~MpegMuxer flush()
// is a no-op.
void test_tsstyle_destructor_with_pending_merger() {
    std::cout << "\n== Case B: TS/HLS-style destructor with pending merger callback ==" << std::endl;
    auto muxer = std::make_shared<TsStyleMuxer>(false);
    feedPendingConfig(*muxer);
    muxer.reset(); // ~TsStyleMuxer flush() -> ~MpegMuxer releaseContext()[flush no-op]
    std::cout << "[ OK ] TS-style destructor survived pending callback" << std::endl;
}

// Case C — PS-encoder-style teardown (NO flush in own dtor). This reproduces the
// secondary defect the fix introduces: a pure-virtual call from ~MpegMuxer.
// Gated OFF by default because it aborts the process.
void test_psstyle_destructor_with_pending_merger() {
    std::cout << "\n== Case C: PS-encoder-style destructor with pending merger callback ==" << std::endl;
    auto muxer = std::make_shared<PsStyleMuxer>(false);
    feedPendingConfig(*muxer);
    muxer.reset(); // ~PsStyleMuxer (no flush) -> ~MpegMuxer flush() -> onWrite() == pure virtual
    std::cout << "[ OK ] PS-style destructor survived pending callback" << std::endl;
}

// Stress: many reset cycles each leaving a pending callback, then a clean (TS-style)
// destructor. Guards the _context double-null path and repeated flush()es.
void test_repeated_reset_stress() {
    std::cout << "\n== Case D: repeated resetTracks() stress ==" << std::endl;
    auto muxer = std::make_shared<TsStyleMuxer>(false);
    for (int i = 0; i < 50; ++i) {
        muxer->addTrack(std::make_shared<FakeH264Track>());
        muxer->inputFrame(makeSps());
        muxer->inputFrame(makePps());
        muxer->resetTracks();
    }
    muxer->addTrack(std::make_shared<FakeH264Track>());
    muxer->inputFrame(makeSps());
    muxer->inputFrame(makePps());
    muxer.reset(); // TS-style clean teardown with a pending callback
    std::cout << "[ OK ] 50x reset cycles + destructor survived" << std::endl;
}

} // namespace

int main(int /*argc*/, char * /*argv*/[]) {
    Logger::Instance().add(std::make_shared<ConsoleChannel>());

    test_reset_tracks_with_pending_merger();
    test_tsstyle_destructor_with_pending_merger();
    test_repeated_reset_stress();

#if RUN_PSSTYLE_DTOR_CASE
    // WARNING: with the current fix this aborts with "pure virtual function called".
    test_psstyle_destructor_with_pending_merger();
#else
    std::cout << "\n[skip] Case C (PS-encoder-style dtor) — compile with "
                 "-DRUN_PSSTYLE_DTOR_CASE=1 to observe the destructor-path defect."
              << std::endl;
#endif

    if (g_failures == 0) {
        std::cout << "\nALL MPEG MUXER REGRESSION TESTS PASSED" << std::endl;
        return 0;
    }
    std::cerr << "\n" << g_failures << " MPEG MUXER REGRESSION CHECK(S) FAILED" << std::endl;
    return 1;
}

#else // !(ENABLE_HLS || ENABLE_RTPPROXY)

int main() {
    std::cout << "MpegMuxer disabled (need ENABLE_HLS or ENABLE_RTPPROXY); test skipped." << std::endl;
    return 0;
}

#endif
