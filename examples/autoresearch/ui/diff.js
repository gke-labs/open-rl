const H = (s = "") => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const rawDiff = (text) => `<pre class="raw-diff">${H(text)}</pre>`;

export function diffHtml(text) {
  if (!text.includes("diff --git ")) return rawDiff(text);
  return parseDiff(text).map(diffFileHtml).join("") || rawDiff(text);
}

function parseDiff(text) {
  const files = [];
  let file, hunk, oldLine = 0, newLine = 0;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("diff --git ")) {
      const match = /^diff --git\s+(?:a\/)?(.+?)\s+(?:b\/)?(.+)$/.exec(line);
      file = { old: cleanDiffPath(match?.[1] || "file"), new: cleanDiffPath(match?.[2] || "file"), hunks: [] };
      files.push(file);
      continue;
    }
    if (!file) continue;
    if (line.startsWith("--- ")) file.old = cleanDiffPath(line.slice(4));
    else if (line.startsWith("+++ ")) file.new = cleanDiffPath(line.slice(4));
    else if (line.startsWith("@@")) {
      const match = /@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?/.exec(line);
      [oldLine, newLine] = match ? [Number(match[1]), Number(match[2])] : [0, 0];
      file.hunks.push(hunk = { content: line, changes: [] });
    } else if (hunk && [" ", "+", "-"].includes(line[0])) {
      const mark = line[0], body = line.slice(1);
      if (mark === " ") hunk.changes.push({ kind: "ctx", old: oldLine++, new: newLine++, text: body });
      if (mark === "+") hunk.changes.push({ kind: "add", new: newLine++, text: body });
      if (mark === "-") hunk.changes.push({ kind: "del", old: oldLine++, text: body });
    }
  }
  return files;
}

function cleanDiffPath(path) {
  const value = path.trim().split("\t")[0];
  return value === "/dev/null" ? value : value.replace(/^[ab]\//, "");
}

function diffFileHtml(file) {
  const changes = file.hunks.flatMap(h => h.changes);
  const deleted = changes.filter(c => c.kind === "del").length, added = changes.filter(c => c.kind === "add").length;
  const name = file.new !== "/dev/null" ? file.new : file.old;
  return `<div class="diff-file"><div class="diff-file-toolbar">
    <button class="diff-toggle" data-action="collapse" aria-label="Collapse file" title="Collapse file"></button>
    <span class="diff-name">${H(name)}</span>
    <span class="diff-counts"><span class="del">-${deleted}</span> <span class="add">+${added}</span></span>
  </div><div class="diff-file-body"><div class="split-diff">${file.hunks.map(diffHunkHtml).join("")}</div></div></div>`;
}

function diffHunkHtml(hunk) {
  return `<div class="diff-hunk"><span>${H(hunk.content)}</span><span>${H(hunk.content)}</span></div>` + pairedChanges(hunk.changes).map(diffRowHtml).join("");
}

function pairedChanges(changes) {
  const rows = [];
  for (let i = 0; i < changes.length;) {
    const change = changes[i];
    if (change.kind === "ctx") {
      rows.push([change.old, change.text, "ctx", change.new, change.text, "ctx"]);
      i++;
      continue;
    }
    const dels = [], adds = [];
    while (i < changes.length && changes[i].kind !== "ctx") (changes[i].kind === "add" ? adds : dels).push(changes[i++]);
    for (let j = 0; j < Math.max(dels.length, adds.length); j++) {
      const left = dels[j], right = adds[j];
      rows.push([left?.old || "", left?.text || "", left ? "del" : "empty", right?.new || "", right?.text || "", right ? "add" : "empty"]);
    }
  }
  return rows;
}

function diffRowHtml([leftNo, left, leftKind, rightNo, right, rightKind]) {
  return `<div class="diff-row"><span class="diff-line-no ${leftKind}">${H(leftNo)}</span><span class="diff-code ${leftKind}">${H(left)}</span><span class="diff-line-no ${rightKind}">${H(rightNo)}</span><span class="diff-code ${rightKind}">${H(right)}</span></div>`;
}
