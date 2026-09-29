// Math question detection + solving, backed by nerdamer.
// Requires nerdamer's core + Algebra + Calculus + Solve add-ons loaded
// BEFORE this file (see index.html: all.min.js covers all of them).
//
// Usage:
//   if (looksLikeMath(text)) {
//     const answer = solveMath(text);
//     if (answer !== null) { ...use answer... }
//   }
//
// Tested against the same cases as the Python math_solver.py; verified with
// nerdamer 1.1.13 via `node -e` before shipping. If nerdamer throws on some
// input, solveMath returns null so the caller can fall back to the chatbot.

const MATH_HINTS = /(\d+\s*[\+\-\*\/\^]\s*\d+|integral|derivative|differentiate|integrate|solve|simplify|factor|limit|d\/dx|∫|√|sqrt|=\s*\?|calculate|compute)/i;

function looksLikeMath(text) {
  return MATH_HINTS.test(text);
}

function solveMath(text) {
  let t = text.toLowerCase().trim();
  t = t.replace(/\?+$/, "").trim();   // strip trailing "?"
  t = t.replace(/=\s*$/, "").trim();  // strip trailing "=" e.g. "2 + 2 ="

  try {
    let m;

    // derivative: "derivative of x^2", "differentiate x^3 + 2x"
    if ((m = t.match(/(derivative|differentiate)\s+(of\s+)?(.+)/))) {
      const expr = m[3];
      const result = nerdamer(`diff(${expr}, x)`).text("fractions");
      return `d/dx [${expr}] = ${result}`;
    }

    // integral: "integrate x^2", "integral of sin(x)"
    if ((m = t.match(/(integrate|integral)\s+(of\s+)?(.+)/))) {
      const expr = m[3];
      const result = nerdamer(`integrate(${expr}, x)`).text("fractions");
      return `∫ ${expr} dx = ${result} + C`;
    }

    // limit: "limit of sin(x)/x as x->0"
    if ((m = t.match(/limit\s+(of\s+)?(.+?)\s+as\s+x\s*->\s*(.+)/))) {
      const expr = m[2], point = m[3];
      const result = nerdamer(`limit(${expr}, x, ${point})`).text("fractions");
      return `lim(x -> ${point}) ${expr} = ${result}`;
    }

    // solve equation: "solve x^2 - 4 = 0" (also accepts no "=0")
    if ((m = t.match(/solve\s+(.+)/))) {
      let eq = m[1];
      if (!eq.includes("=")) eq += "=0";
      const sols = nerdamer.solve(eq, "x");
      return `Solving ${eq} gives x = ${sols.toString()}`;
    }

    // simplify / factor: "simplify (x^2-1)/(x-1)", "factor x^2 - 4"
    if ((m = t.match(/(simplify|factor)\s+(.+)/))) {
      const fn = m[1], expr = m[2];
      const result = nerdamer(`${fn}(${expr})`).text("fractions");
      return `${fn[0].toUpperCase() + fn.slice(1)}ing ${expr} gives ${result}`;
    }

    // plain arithmetic / expression: "2 + 2", "sqrt(144)", "5 * (3 + 2)"
    const result = nerdamer(t).evaluate().text("fractions");
    return `${t} = ${result}`;

  } catch (e) {
    console.error("math_solver.js: could not parse as math:", e);
    return null; // not parseable; caller should fall back to the chatbot
  }
}