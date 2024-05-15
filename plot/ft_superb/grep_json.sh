find ~/datasets/superb/results/*gpu-*-nano*-13w*/*.json -exec jq 'select(.["metric-value"] != -1)' -c {} \; > superb.partial.json
find ~/datasets/superb/results/64gpu*/*.json -exec jq 'select(.["metric-value"] != -1)' -c {} \; > fairseq.partial.json
