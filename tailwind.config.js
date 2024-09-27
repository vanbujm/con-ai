/* eslint-disable @typescript-eslint/no-require-imports,no-undef */
const defaultTheme = require("tailwindcss/defaultTheme")

const disabledCss = {
  "code::before": false,
  "code::after": false,
  "pre": false,
  "code": false,
  "pre code": false,
}

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./post.md"],
  theme: {
    fontFamily: {
      sans: ["Nunito", ...defaultTheme.fontFamily.sans],
      mono: ["Fira Code", ...defaultTheme.fontFamily.mono],
    },
    extend: {
      typography: {
        "DEFAULT": { css: disabledCss },
        "sm": { css: disabledCss },
        "lg": { css: disabledCss },
        "xl": { css: disabledCss },
        "2xl": { css: disabledCss },
      },
    },
  },
  daisyui: {
    themes: ["forest"],
  },
  typography: {
    default: {
      css: {
        "pre": false,
        "code": false,
        "pre code": false,
        "code::before": false,
        "code::after": false,
      },
    },
  },
  plugins: [require("@tailwindcss/typography"), require("daisyui")],
}
