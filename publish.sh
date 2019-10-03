#!/bin/bash -e

if [ -n "$(git status --porcelain)" ]; then
    echo "**** Untracked changes, not publishing ***"
    git status
    echo "**** Untracked changes, not publishing ***"
    exit 1
fi

diffs="$(git diff origin/master)"

if [ -n "$diffs" ]; then
    echo "**** Unpushed changes, not publishing ***"
    echo "$diffs"
    echo "**** Unpushed changes, not publishing ***"
    exit 1
fi

echo "See https://twine.readthedocs.io/en/latest/ for more info"

rm -rf dist
python3 setup.py bdist_wheel
twine upload dist/*
