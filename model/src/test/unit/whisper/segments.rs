use crate::whisper::{WhisperTokenizer, split_into_segments, window_seek};

fn tokenizer() -> WhisperTokenizer {
    WhisperTokenizer::from_hub(true, 99).unwrap()
}

#[test]
fn unfinished_timestamp_tail_is_not_emitted() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.extend([timestamp + 50, timestamp + 50]);
    tokens.extend(tokenizer.encode(" unfinished"));

    let segments = split_into_segments(&tokens, &tokenizer, 30.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hello");
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 1.0);
}

#[test]
fn segment_without_timestamps_spans_real_window() {
    let tokenizer = tokenizer();
    let tokens = tokenizer.encode(" hello");
    let segments = split_into_segments(&tokens, &tokenizer, 2.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 0.0);
    assert_eq!(segments[0].end, 2.5);
}

#[test]
fn last_timestamp_limits_unpaired_segment() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.push(timestamp + 75);
    tokens.extend(tokenizer.encode(" tail"));

    let segments = split_into_segments(&tokens, &tokenizer, 4.0);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].end, 1.5);
}

#[test]
fn timestamp_segments_are_clipped_to_real_audio_extent() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp + 50];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.extend([timestamp + 500, timestamp + 500]);
    tokens.extend(tokenizer.encode(" beyond"));
    tokens.push(timestamp + 600);

    let segments = split_into_segments(&tokens, &tokenizer, 2.5);
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, 1.0);
    assert_eq!(segments[0].end, 2.5);
}

// ─── Window seek ────────────────────────────────────────────────────────────

/// Without a completed timestamp pair the stream is one segment covering the
/// whole window, so the read head moves past all of it.
#[test]
fn seek_without_timestamp_pairs_advances_the_whole_window() {
    let tokenizer = tokenizer();
    let mut tokens = vec![tokenizer.timestamp_begin()];
    tokens.extend(tokenizer.encode(" hello"));

    assert_eq!(window_seek(&tokens, &tokenizer, 30.0), 30.0);
    assert_eq!(window_seek(&[], &tokenizer, 30.0), 30.0, "an empty stream cannot limit the seek");
}

/// A stream that ended mid-segment resumes at the last completed pair, so the
/// unfinished tail is decoded again with its audio intact.
#[test]
fn seek_stops_at_the_last_completed_pair_when_the_tail_is_unfinished() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.extend([timestamp + 50, timestamp + 50]);
    tokens.extend(tokenizer.encode(" second"));
    tokens.extend([timestamp + 100, timestamp + 100]);
    tokens.extend(tokenizer.encode(" unfinished"));

    assert_eq!(window_seek(&tokens, &tokenizer, 30.0), 2.0, "the last pair, not the first");
    assert_eq!(window_seek(&tokens, &tokenizer, 1.5), 1.5, "a pair past the window's end is clamped to it");
}

/// A lone trailing timestamp means nothing was spoken after it: there is no
/// unfinished tail to re-decode, so the head moves past the whole window.
#[test]
fn seek_advances_the_whole_window_when_the_stream_ends_in_a_lone_timestamp() {
    let tokenizer = tokenizer();
    let timestamp = tokenizer.timestamp_begin();
    let mut tokens = vec![timestamp];
    tokens.extend(tokenizer.encode(" hello"));
    tokens.extend([timestamp + 50, timestamp + 50]);
    tokens.extend(tokenizer.encode(" tail"));
    tokens.push(timestamp + 100);

    assert_eq!(window_seek(&tokens, &tokenizer, 30.0), 30.0);
}
