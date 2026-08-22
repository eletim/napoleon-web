# Card Design Vendor Runtime

These files support the selected production card renderer. Issue #392 comparison-only SVGCards fixtures were removed during Issue #418 cleanup.

## cardmeister

- Source: https://github.com/cardmeister/cardmeister.github.io
- License: Unlicense, per the upstream README.
- npm: no npm package is used.
- Vendor asset: `cardmeister/elements.cardmeister.full.js`.
- Runtime usage: production and retained mocks load this custom element through `CardmeisterPlayingCard`.

The app renders `<playing-card>` custom elements with both `suitcolor` and `rankcolor` set from the shared four-suit theme:

- Spades: `#111827`
- Hearts: `#dc2626`
- Diamonds: `#2563eb`
- Clubs: `#15803d`
