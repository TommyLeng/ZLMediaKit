/*
 * Copyright (c) 2016-present The ZLMediaKit project authors. All Rights Reserved.
 *
 * This file is part of ZLMediaKit(https://github.com/ZLMediaKit/ZLMediaKit).
 *
 * Use of this source code is governed by MIT-like license that can be found in the
 * LICENSE file in the root of the source tree. All contributing project authors
 * may be found in the AUTHORS file in the root of the source tree.
 */

// ───────────────────────────── What this tests ──────────────────────────────
//
// Regression test for the zero-connection snapshot path. It drives the real API
// entry point, FFmpegSnap::makeSnap(), so the whole chain is exercised:
//
//   FFmpegSnap::makeSnap(async, "rtmp://127.0.0.1/...")
//     -> isLocalUrl()        local host? -> take the internal path
//     -> makeSnapInternal()  find stream -> flushGop() -> FFmpegDecoder -> saveFrame
//
// A tiny MP4 fixture is loaded into a local MediaSource; once its GOP cache is
// populated, makeSnap() is called and the decoded JPEG is asserted valid.
//
// Both async=0 and async=1 are tested: a local URL must take makeSnapInternal()
// either way. This doubles as proof of the routing — the test never starts an RTMP
// server, so if async=1 wrongly fell through to makeSnapAsync() (which opens a
// MediaPlayer to rtmp://127.0.0.1) it would fail to connect; passing means the
// internal in-memory path was taken.
//
// A third case points at a non-existent local stream to exercise the fallback: when
// makeSnapInternal can't serve the request it defers to the normal path, which must
// run to completion and report failure (not hang).
//
// ─────────────────────────────── How to run ─────────────────────────────────
//
// Build an FFmpeg-enabled tree with tests turned on:
//
//     mkdir build && cd build
//     cmake -DENABLE_FFMPEG=ON -DENABLE_MP4=ON -DENABLE_TESTS=ON ..
//     make test_getsnap -j
//
// Run it (the binary sits next to the other test binaries, e.g.
// release/linux/Release/ or release/darwin/Release/):
//
//     ./test_getsnap                 # uses the bundled test_snap_fixture.mp4
//     ./test_getsnap /path/to/x.mp4  # or point it at your own MP4
//
// Exit code 0 = pass, non-zero = fail. On success the snapshot is written to
// test_snap_output.jpg next to the binary.
//
// Without -DENABLE_FFMPEG=ON (and -DENABLE_MP4=ON) the test compiles to a no-op
// that prints a skip notice and returns 0.
//
// The bundled fixture is a 5s 640x360 H.264 clip with a key frame every second.
// To regenerate it:
//     ffmpeg -f lavfi -i color=c=red:size=640x360:rate=30:duration=1 \
//            -f lavfi -i color=c=green:size=640x360:rate=30:duration=1 \
//            -f lavfi -i color=c=blue:size=640x360:rate=30:duration=1 \
//            -filter_complex "[0][1][2]concat=n=3:v=1:a=0[v]" -map "[v]" \
//            -c:v libx264 -profile:v baseline -pix_fmt yuv420p -g 30 -keyint_min 30 \
//            -sc_threshold 0 tests/test_snap_fixture.mp4
// ─────────────────────────────────────────────────────────────────────────────

#if defined(ENABLE_FFMPEG) && defined(ENABLE_MP4)

#include <unistd.h>
#include <atomic>
#include <iostream>
#include "Util/File.h"
#include "Util/logger.h"
#include "Util/util.h"
#include "Util/TimeTicker.h"
#include "Thread/semaphore.h"
#include "Common/config.h"
#include "Common/MediaSource.h"
#include "Common/MultiMediaSourceMuxer.h"
#include "Poller/EventPoller.h"
#include "Record/MP4Reader.h"
#include "Extension/Frame.h"
#include "FFmpegSource.h" // FFmpegSnap — compiled into this test target via tests/CMakeLists.txt

using namespace std;
using namespace toolkit;
using namespace mediakit;

namespace {

// A JPEG always starts with the SOI marker 0xFF 0xD8.
bool isJpeg(const string &path) {
    auto data = File::loadFile(path.data());
    return data.size() >= 2 && (uint8_t)data[0] == 0xFF && (uint8_t)data[1] == 0xD8;
}

// Wait until the local stream exists, its video track is ready, and the GOP cache holds
// at least one video frame — i.e. exactly the preconditions makeSnapInternal() checks.
// This guarantees makeSnap() takes the internal (zero-connection) path instead of falling
// back to an external FFmpeg process (which the test environment has no server for).
bool waitForGopReady(const string &app, const string &stream, int timeout_ms) {
    Ticker ticker;
    while (ticker.elapsedTime() < timeout_ms) {
        usleep(100 * 1000);

        auto src = MediaSource::find(DEFAULT_VHOST, app, stream, false);
        if (!src) {
            continue;
        }
        auto muxer = src->getMuxer();
        if (!muxer) {
            continue;
        }
        bool has_video_track = false;
        for (auto &t : muxer->getTracks(true)) {
            if (t->getTrackType() == TrackVideo) {
                has_video_track = true;
                break;
            }
        }
        if (!has_video_track) {
            continue;
        }
        size_t video_frames = 0;
        muxer->flushGop([&](const Frame::Ptr &f) {
            if (f->getTrackType() == TrackVideo) {
                ++video_frames;
            }
        });
        if (video_frames > 0) {
            return true;
        }
    }
    return false;
}

struct SnapResult {
    bool cb_fired = false;  // false => makeSnap hung (watchdog tripped)
    bool ok = false;        // the success flag passed to the callback
    bool valid_jpeg = false;// a non-empty JPEG actually landed on disk
    string err;
};

// Calls the real API entry point with the given async flag and blocks until the callback
// fires (makeSnapInternal decodes on a WorkThreadPool). A watchdog guarantees we report a
// non-fire rather than hang forever.
SnapResult runSnap(bool async, const string &url, const string &save_path,
                   const EventPoller::Ptr &poller) {
    File::delete_file(save_path.data());

    semaphore sem;
    std::atomic<bool> cb_fired{false};
    SnapResult result;

    FFmpegSnap::makeSnap(async, url, save_path, 10.0f,
        [&](bool success, const string &msg) {
            result.ok = success;
            result.err = msg;
            cb_fired = true;
            sem.post();
        });

    auto watchdog = poller->doDelayTask(20000, [&]() {
        if (!cb_fired) {
            result.err = "callback never fired (watchdog timeout)";
            sem.post();
        }
        return 0;
    });
    sem.wait();
    watchdog->cancel();

    result.cb_fired = cb_fired;
    if (result.cb_fired && result.ok) {
        result.valid_jpeg = File::fileSize(save_path.data()) > 0 && isJpeg(save_path);
    }
    return result;
}

} // namespace

int main(int argc, char *argv[]) {
    Logger::Instance().add(std::make_shared<ConsoleChannel>());
    Logger::Instance().setWriter(std::make_shared<AsyncLogWriter>());

    const string fixture = argc > 1 ? argv[1] : (exeDir() + "test_snap_fixture.mp4");
    if (!File::fileExist(fixture.data())) {
        ErrorL << "fixture not found: " << fixture << " (pass an MP4 path as argv[1])";
        return 1;
    }
    const string save_path = exeDir() + "test_snap_output.jpg";
    File::delete_file(save_path.data());

    // Register an eager local stream so MediaSource::find() can locate it without a
    // reader attached (rtmp_demand = false). Mirrors a real pushed/pulled local stream.
    ProtocolOption option;
    option.enable_rtmp = true;
    option.rtmp_demand = false;
    option.enable_audio = false;

    auto poller = EventPollerPool::Instance().getPoller();
    auto tuple = MediaTuple{DEFAULT_VHOST, "live", "snap_test", ""};
    auto reader = std::make_shared<MP4Reader>(tuple, fixture, option, poller);
    // ref_self = true keeps the reader alive; file_repeat = true loops the short fixture
    // so the stream stays up while we wait for the GOP cache to fill.
    reader->startReadMP4(100, true, true);

    if (!waitForGopReady("live", "snap_test", 15000)) {
        ErrorL << "FAIL: local stream / GOP cache not ready in time";
        return 1;
    }

    // Case 1 & 2 — happy path: a local 127.0.0.1 URL must route through isLocalUrl() ->
    // makeSnapInternal() (the zero-connection path) for BOTH async values. See the header
    // note on why async=1 passing proves the internal path was taken.
    const string url = "rtmp://127.0.0.1/live/snap_test";
    for (bool async : {false, true}) {
        auto r = runSnap(async, url, save_path, poller);
        if (!r.cb_fired) {
            ErrorL << "FAIL: makeSnap(async=" << async << ") never invoked its callback: " << r.err;
            return 1;
        }
        if (!r.ok || !r.valid_jpeg) {
            ErrorL << "FAIL: makeSnap(async=" << async << ") produced no valid snapshot: " << r.err;
            return 1;
        }
        InfoL << "PASS: makeSnap(async=" << async << ") wrote a valid JPEG via the local internal path";
    }

    // Case 3 — fallback path: a local URL for a stream that does not exist must NOT hang.
    // makeSnapInternal finds nothing -> on_fallback() -> makeSnapSync() spawns an external
    // FFmpeg that cannot connect/find the stream -> the callback fires with failure. We
    // assert the callback fired (the fallback chain ran to completion) and reported failure.
    {
        auto r = runSnap(false, "rtmp://127.0.0.1/live/no_such_stream_xyz", save_path, poller);
        if (!r.cb_fired) {
            ErrorL << "FAIL: fallback path hung — callback never fired for a missing stream";
            return 1;
        }
        if (r.ok) {
            ErrorL << "FAIL: snapshot unexpectedly succeeded for a non-existent stream";
            return 1;
        }
        InfoL << "PASS: missing-stream request fell back and reported failure without hanging";
    }

    reader->stopReadMP4();
    return 0;
}

#else // !(ENABLE_FFMPEG && ENABLE_MP4)

#include <iostream>

int main(int argc, char *argv[]) {
    std::cout << "test_getsnap skipped: build with -DENABLE_FFMPEG=ON -DENABLE_MP4=ON to run it" << std::endl;
    return 0;
}

#endif
