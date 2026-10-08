#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# The 0.0.4 oracle for voaice.rs's Ogg/Opus reader: opus-tools 0.2 (opusinfo, opusdec), libopus 1.4 and libogg 1.3.5
# on mindX production, reached read-only over ssh (BatchMode); all work in one scratch dir there, removed after.
#
#   testing/opus/oracle.sh record [host]   build the corpus, encode the speech on the host, fetch it back, make the
#                                          adversarial files, run the reference on everything; writes testing/opus/
#                                          files/*.opus, files.sha256, mutations.jsonl, reference.jsonl, reference.meta.json
#   testing/opus/oracle.sh check [host]    run the reference again on the pinned files (and the adversarial files made
#                                          again from them) and require the same answers as recorded
#
# host defaults to root@168.231.126.58. Nothing is installed, no service is touched; the scratch dir is
# /tmp/voaice_oracle_004 on the host.
set -euo pipefail
mode=${1:-check}
host=${2:-root@168.231.126.58}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
remote=/tmp/voaice_oracle_004
ssh_() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" "$@"; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# opusenc settings: name | input | options (every file is encoded with a fixed serial, so the bytes are reproducible)
encodes=(
  "e_jfk_m_2.5ms_24k|jfk|--framesize 2.5 --bitrate 24"
  "e_jfk_m_5ms_24k|jfk|--framesize 5 --bitrate 24"
  "e_jfk_m_10ms_24k|jfk|--framesize 10 --bitrate 24"
  "e_jfk_m_20ms_24k|jfk|--framesize 20 --bitrate 24"
  "e_jfk_m_40ms_24k|jfk|--framesize 40 --bitrate 24"
  "e_jfk_m_60ms_24k|jfk|--framesize 60 --bitrate 24"
  "e_jfk_m_20ms_6k|jfk|--bitrate 6"
  "e_jfk_m_20ms_12k_delay0|jfk|--bitrate 12 --max-delay 0"
  "e_jfk_x3_m_6k_comp0|jfk_x3|--bitrate 6 --comp 0"
  "e_chirp_64k_cbr|chirp|--bitrate 64 --hard-cbr"
  "e_noise_loud_64k|noise_loud|--bitrate 64"
  "e_silence_24k|silence|--bitrate 24"
  "e_short_24k|short|--bitrate 24"
  "e_odd_len_24k|odd_len|--bitrate 24"
  "e_min_len_24k|min_len|--bitrate 24"
  "e_stereo_20ms_48k|stereo|--bitrate 48"
  "e_stereo_60ms_32k|stereo|--framesize 60 --bitrate 32"
  "e_stereo_downmix_24k|stereo|--downmix-mono --bitrate 24"
  "e_six_family1_96k|six|--bitrate 96"
  "e_picture_24k|short|--bitrate 24 --picture art.png"
  "e_comments_24k|short|--bitrate 24 --title Ünïcødé --artist voaice.rs --comment LANGUAGE=en --comment X=a=b --padding 0"
)

if [ "$mode" = record ]; then
  cargo build --release -q --manifest-path "$root/streamair/Cargo.toml" --example opus_corpus --bin streamair
  mkdir -p "$work/files" "$work/in"
  for t in 0.001 1 2.5 7.3333 60; do "$root/streamair/target/release/streamair" silence "$t" "$work/files/sa_silence_$t.opus" >/dev/null; done
  "$root/streamair/target/release/examples/opus_corpus" "$work/files" >/dev/null
  python3 "$here/make_inputs.py" "$work/in"
  ssh_ "rm -rf $remote && mkdir -p $remote/in $remote/files"
  scp -q "$work"/in/* "$host:$remote/in/"
  scp -q "$here/reference.py" "$host:$remote/"
  cmds=""
  for e in "${encodes[@]}"; do
    IFS='|' read -r name input opts <<<"$e"
    cmds+="(cd $remote/in && opusenc --quiet --serial 4004 $opts $input.wav $remote/files/$name.opus) && "
  done
  ssh_ "$cmds true"
  scp -q "$host:$remote/files/*.opus" "$work/files/"
  python3 "$here/mutate.py" "$work/files" "$work/mut" > "$work/mutations.jsonl"
  ssh_ "mkdir -p $remote/mut"
  scp -q "$work"/files/sa_*.opus "$host:$remote/files/"
  scp -q "$work"/mut/*.opus "$host:$remote/mut/"
  ssh_ "cd $remote && python3 reference.py files/*.opus mut/*.opus" > "$work/reference.jsonl"
  {
    printf '{"date": "%s", "host_cpu": "%s", ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(ssh_ "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs")"
    printf '"reference": %s, ' "$(ssh_ "cd $remote && python3 reference.py --versions")"
    printf '"packages": "%s", ' "$(ssh_ "dpkg-query -W -f='\${Package} \${Version}; ' opus-tools libopus0 libogg0")"
    printf '"encodes": ['
    sep=""; for e in "${encodes[@]}"; do printf '%s"%s"' "$sep" "$e"; sep=", "; done
    printf ']}\n'
  } > "$work/reference.meta.json"
  ssh_ "rm -rf $remote"
  rm -rf "$here/files" && mkdir -p "$here/files"
  cp "$work"/files/*.opus "$here/files/"
  (cd "$here/files" && sha256sum *.opus) > "$here/files.sha256"
  cp "$work/mutations.jsonl" "$work/reference.jsonl" "$work/reference.meta.json" "$here/"
  echo "recorded $(ls "$here/files" | wc -l) files and $(wc -l < "$here/mutations.jsonl") adversarial files; reference.jsonl $(wc -l < "$here/reference.jsonl") lines"
  exit 0
fi

[ "$mode" = check ] || { echo "usage: oracle.sh record|check [host]"; exit 2; }
(cd "$here/files" && sha256sum -c --quiet ../files.sha256)
python3 "$here/mutate.py" "$here/files" "$work/mut" > "$work/mutations.jsonl"
cmp -s "$work/mutations.jsonl" "$here/mutations.jsonl" || { echo "FAIL: mutate.py made different adversarial files"; exit 1; }
ssh_ "rm -rf $remote && mkdir -p $remote/files $remote/mut"
scp -q "$here/reference.py" "$host:$remote/"
scp -q "$here"/files/*.opus "$host:$remote/files/"
scp -q "$work"/mut/*.opus "$host:$remote/mut/"
ssh_ "cd $remote && python3 reference.py files/*.opus mut/*.opus" > "$work/reference.jsonl"
ssh_ "rm -rf $remote"
if cmp -s "$work/reference.jsonl" "$here/reference.jsonl"; then
  echo "reference re-run on $host: $(wc -l < "$work/reference.jsonl") / $(wc -l < "$here/reference.jsonl") answers identical to the record"
else
  diff <(cut -c1-300 "$here/reference.jsonl") <(cut -c1-300 "$work/reference.jsonl") | head -20
  echo "FAIL: the reference's answers changed"; exit 1
fi
