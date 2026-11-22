#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

build_report() {
    echo "==> Compiling Report..."
    cd "$DIR/report"
    pdflatex -interaction=nonstopmode report.tex > /dev/null
    pdflatex -interaction=nonstopmode report.tex > /dev/null
    echo "✓ Report built: reports/report/report.pdf"
}

build_presentation() {
    echo "==> Compiling Presentation..."
    cd "$DIR/presentation"
    pdflatex -interaction=nonstopmode presentation.tex > /dev/null
    pdflatex -interaction=nonstopmode presentation.tex > /dev/null
    echo "✓ Presentation built: reports/presentation/presentation.pdf"
}

clean() {
    echo "==> Cleaning auxiliary files..."
    rm -f "$DIR"/report/*.aux "$DIR"/report/*.log "$DIR"/report/*.out "$DIR"/report/*.toc "$DIR"/report/*.lof "$DIR"/report/*.lot
    rm -f "$DIR"/presentation/*.aux "$DIR"/presentation/*.log "$DIR"/presentation/*.out "$DIR"/presentation/*.toc "$DIR"/presentation/*.nav "$DIR"/presentation/*.snm
    echo "✓ Cleaned."
}

case "${1:-all}" in
    report)
        build_report
        ;;
    presentation|pres)
        build_presentation
        ;;
    clean)
        clean
        ;;
    all)
        build_report
        build_presentation
        ;;
    *)
        echo "Usage: ./compile.sh [report|presentation|clean|all]"
        exit 1
        ;;
esac
