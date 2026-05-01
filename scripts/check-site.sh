#!/usr/bin/env bash
set -euo pipefail

DOCS_DIR="${DOCS_DIR:-docs}"
ERRORS=0
CHECKS_PASSED=0

# Collect all HTML files in docs/ root (excluding docs/data/)
# Use while-read loop for bash 3.x compatibility (macOS ships bash 3.2)
PAGES=()
while IFS= read -r f; do
  PAGES+=("$f")
done < <(find "$DOCS_DIR" -maxdepth 1 -name "*.html" | sort)

if [[ ${#PAGES[@]} -eq 0 ]]; then
  echo "ERROR: No HTML files found in $DOCS_DIR/" >&2
  exit 1
fi

echo "=== JobSentinel Site Consistency Check ==="
echo "Checking ${#PAGES[@]} pages: $(printf '%s ' "${PAGES[@]##*/}")"
echo ""

# ----------------------------------------------
# Helpers
# ----------------------------------------------
fail() {
  echo "FAIL: $*"
  ERRORS=$((ERRORS + 1))
}

pass() {
  echo "PASS: $*"
  CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

# ----------------------------------------------
# 1. Nav link consistency
#    Every page must link to all 5 core pages.
#    Self-links are skipped (a page need not link to itself).
#    index.html target is also satisfied by href="/" or href="./".
# ----------------------------------------------
echo "--- 1. Nav link consistency ---"
REQUIRED_NAV=("index.html" "analyze.html" "gallery.html" "jobs.html" "demo.html")
NAV_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  for link in "${REQUIRED_NAV[@]}"; do
    # Skip self-referential check
    [[ "$name" == "$link" ]] && continue
    # For index.html target, also accept href="/" or href="./"
    if [[ "$link" == "index.html" ]]; then
      if grep -qE 'href="(index\.html|/|\./)' "$page" 2>/dev/null; then
        continue
      fi
    else
      if grep -q "href=\"${link}\"" "$page" 2>/dev/null; then
        continue
      fi
    fi
    fail "$name: missing nav link to $link"
    NAV_ERRORS=$((NAV_ERRORS + 1))
  done
done
[[ $NAV_ERRORS -eq 0 ]] && pass "All pages have required nav links"
echo ""

# ----------------------------------------------
# 2. Required meta tags
# ----------------------------------------------
echo "--- 2. Required meta tags ---"
META_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  if ! grep -qi '<meta[[:space:]][^>]*name="description"' "$page" 2>/dev/null; then
    fail "$name: missing <meta name=\"description\">"
    META_ERRORS=$((META_ERRORS + 1))
  fi
  if ! grep -qi '<meta[[:space:]][^>]*name="viewport"' "$page" 2>/dev/null; then
    fail "$name: missing <meta name=\"viewport\">"
    META_ERRORS=$((META_ERRORS + 1))
  fi
done
[[ $META_ERRORS -eq 0 ]] && pass "All pages have required meta tags (description, viewport)"
echo ""

# ----------------------------------------------
# 3. Footer copyright year
# ----------------------------------------------
echo "--- 3. Footer copyright ---"
COPYRIGHT_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  if ! grep -q "2026" "$page" 2>/dev/null; then
    fail "$name: missing copyright year 2026"
    COPYRIGHT_ERRORS=$((COPYRIGHT_ERRORS + 1))
  fi
done
[[ $COPYRIGHT_ERRORS -eq 0 ]] && pass "All pages contain copyright year 2026"
echo ""

# ----------------------------------------------
# 4. style.css link
# ----------------------------------------------
echo "--- 4. style.css link ---"
CSS_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  if ! grep -q 'href="style\.css"' "$page" 2>/dev/null; then
    fail "$name: missing link to style.css"
    CSS_ERRORS=$((CSS_ERRORS + 1))
  fi
done
[[ $CSS_ERRORS -eq 0 ]] && pass "All pages link to style.css"
echo ""

# ----------------------------------------------
# 5. Favicon (data:image/svg)
# ----------------------------------------------
echo "--- 5. Favicon ---"
FAVICON_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  if ! grep -q 'data:image/svg' "$page" 2>/dev/null; then
    fail "$name: missing favicon (data:image/svg)"
    FAVICON_ERRORS=$((FAVICON_ERRORS + 1))
  fi
done
[[ $FAVICON_ERRORS -eq 0 ]] && pass "All pages have a favicon (data:image/svg)"
echo ""

# ----------------------------------------------
# 6. Internal link integrity
#    For every href="*.html" that is not an absolute URL,
#    verify the target file exists in DOCS_DIR.
# ----------------------------------------------
echo "--- 6. Internal link integrity ---"
LINK_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  while IFS= read -r target; do
    # Strip any fragment
    target_base="${target%%#*}"
    [[ -z "$target_base" ]] && continue
    # Skip external URLs
    [[ "$target_base" =~ ^https?:// ]] && continue
    [[ "$target_base" =~ ^// ]] && continue
    # Only check .html targets
    [[ "$target_base" != *.html ]] && continue
    target_path="$DOCS_DIR/$target_base"
    if [[ ! -f "$target_path" ]]; then
      fail "$name: broken link -> $target_base (file not found)"
      LINK_ERRORS=$((LINK_ERRORS + 1))
    fi
  done < <(grep -oE 'href="[^"]+"' "$page" 2>/dev/null | sed 's/href="//;s/"//')
done
[[ $LINK_ERRORS -eq 0 ]] && pass "All internal .html links resolve to existing files"
echo ""

# ----------------------------------------------
# 7. aria-label on nav logo
# ----------------------------------------------
echo "--- 7. aria-label on nav logo ---"
ARIA_LOGO_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  if ! grep -q 'class="nav-logo"[^>]*aria-label\|aria-label[^>]*class="nav-logo"' "$page" 2>/dev/null; then
    fail "$name: nav logo (.nav-logo) missing aria-label"
    ARIA_LOGO_ERRORS=$((ARIA_LOGO_ERRORS + 1))
  fi
done
[[ $ARIA_LOGO_ERRORS -eq 0 ]] && pass "All pages have aria-label on nav logo"
echo ""

# ----------------------------------------------
# 8. Mobile toggle aria-expanded
#    Only checked on pages that have a mobile toggle button.
# ----------------------------------------------
echo "--- 8. Mobile toggle aria-expanded ---"
ARIA_EXP_ERRORS=0
for page in "${PAGES[@]}"; do
  name=$(basename "$page")
  # Skip pages without a mobile toggle entirely
  if ! grep -q 'nav-mobile-toggle' "$page" 2>/dev/null; then
    continue
  fi
  # The toggle button line must include aria-expanded
  if ! grep 'nav-mobile-toggle' "$page" 2>/dev/null | grep -q 'aria-expanded'; then
    fail "$name: nav-mobile-toggle button missing aria-expanded"
    ARIA_EXP_ERRORS=$((ARIA_EXP_ERRORS + 1))
  fi
done
[[ $ARIA_EXP_ERRORS -eq 0 ]] && pass "All mobile toggle buttons have aria-expanded"
echo ""

# ----------------------------------------------
# Summary
# ----------------------------------------------
echo "==========================================="
if [[ $ERRORS -eq 0 ]]; then
  echo "All checks passed (${CHECKS_PASSED} check groups, ${#PAGES[@]} pages)"
  exit 0
else
  echo "$ERRORS issue(s) found across ${#PAGES[@]} pages"
  exit 1
fi
