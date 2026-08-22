# Card Design Comparison Fixtures

These files are used only by `/mock/card-design` for Issue #392 visual comparison.

## SVGCards

- Source: https://github.com/saulspatz/SVGCards
- License: Public Domain, per the upstream README.
- npm: not used for this fixture.
- Vendor assets: 11 PNG cards from each of `Decks/Vertical4/pngs`, `Decks/Horizontal4/pngs`, and `Decks/Accessible/Horizontal/pngs`.
- Cards included: A spades/clubs/hearts/diamonds, 10 spades/clubs/hearts/diamonds, J spades, Q hearts, K diamonds.

The upstream project also provides SVG files. The individual court-card SVG files are large, so this mock vendors the upstream PNG renderings to keep the comparison fixture small.

## cardmeister

- Source: https://github.com/cardmeister/cardmeister.github.io
- License: Unlicense, per the upstream README.
- npm: no npm package is used.
- Vendor asset: `elements.cardmeister.full.js`.

The mock renders `<playing-card>` custom elements with `suitcolor` and `rankcolor` set to spades black, hearts red, diamonds blue, clubs green.
