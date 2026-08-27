/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Deep technical greys - the workspace chrome.
        ink: {
          950: '#06080f',
          900: '#0a0e1a',
          850: '#0f1424',
          800: '#141b2e',
          750: '#1a2238',
          700: '#212c46',
          600: '#2d3a58',
          500: '#3d4c6e',
        },
        // Blueprint cyan - the accent that carries the drafting metaphor.
        blueprint: {
          400: '#4cc9f0',
          500: '#22b8e6',
          600: '#0e9bc9',
        },
        // Signal colours for pipeline state.
        signal: {
          ok: '#34d399',
          warn: '#fbbf24',
          err: '#f87171',
          run: '#60a5fa',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Consolas', 'monospace'],
      },
      backgroundImage: {
        grid: 'linear-gradient(rgba(76,201,240,.07) 1px, transparent 1px), linear-gradient(90deg, rgba(76,201,240,.07) 1px, transparent 1px)',
      },
      backgroundSize: { grid: '32px 32px' },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(.9)', opacity: '.7' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
      },
      animation: {
        'fade-up': 'fade-up .4s ease-out both',
        'pulse-ring': 'pulse-ring 1.6s ease-out infinite',
      },
    },
  },
  plugins: [],
}
