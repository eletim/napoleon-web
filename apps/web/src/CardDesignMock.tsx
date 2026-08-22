import type { CSSProperties } from "react";
import {
  CardDesignPrototypeCard,
  cardDesignCardHeight,
  cardDesignComparisonRanks,
  cardDesignConfig,
  cardDesignExposureLeft,
  cardDesignExposureOffset,
  cardDesignIdentificationGuideX,
  cardDesignOverlapWidth,
  cardDesignSuitLabels,
  cardDesignSuitOrder,
  cardDesignSuitSymbols,
  createCardDesignDeck
} from "./CardDesignCard";
import {
  CardmeisterPlayingCard,
  useCardmeisterScript
} from "./CardmeisterPlayingCard";
import {
  mockPlayingCardComponent,
  mockPlayingCardComponentName,
  type MockPlayingCard,
  type MockPlayingCardSuit
} from "./mockPlayingCardAdapter";
import { resolveAppPath } from "./appPath";
import "./CardDesignMock.css";

type CandidateCardSource = "cardmeister" | "letele" | "svgcards";

interface CardDesignCandidate {
  assetFolder?: string;
  aspectRatio: number;
  id: string;
  license: string;
  npm: string;
  selected?: boolean;
  source: CandidateCardSource;
  sourceLabel: string;
  summary: string;
  title: string;
  vendor: string;
}

interface CandidateComparisonMode {
  cards: readonly MockPlayingCard[];
  className: string;
  displayWidth: number;
  metric: string;
  title: string;
  sourceWidth: number;
  clipped?: boolean;
}

const focusCards = createCardDesignDeck(cardDesignComparisonRanks);
const fullDeckCards = createCardDesignDeck();
const tenExposureCards = cardDesignSuitOrder.map((suit) => ({ rank: "10", suit })) satisfies readonly MockPlayingCard[];
const faceComparisonCards = createCardDesignDeck(["J", "Q", "K"]);
const fourColorComparisonCards: readonly MockPlayingCard[] = [
  { rank: "A", suit: "spades" },
  { rank: "A", suit: "clubs" },
  { rank: "A", suit: "hearts" },
  { rank: "A", suit: "diamonds" },
  { rank: "10", suit: "spades" },
  { rank: "10", suit: "clubs" },
  { rank: "10", suit: "hearts" },
  { rank: "10", suit: "diamonds" },
  { rank: "J", suit: "spades" },
  { rank: "Q", suit: "hearts" },
  { rank: "K", suit: "diamonds" }
];
const currentTrickComparisonCards = fourColorComparisonCards.slice(8);
const currentTrickEquivalent = { width: 342.939, height: 480.115 };
const selfHandEquivalent = { width: 118.154, height: 165.415 };
const riverEquivalent = { width: 126.675, visibleHeight: 44.336 };
const leteleAspectRatio = 7 / 5;
const svgCardsPngAspectRatio = 113 / 75;

const cardDesignCandidates = [
  {
    id: "letele",
    title: "Current @letele",
    source: "letele",
    sourceLabel: "@letele/playing-cards",
    license: "CC0-1.0",
    npm: "installed npm dependency",
    vendor: "none",
    aspectRatio: leteleAspectRatio,
    summary: "Baseline two-color deck; court art is familiar but club/spade and heart/diamond depend mostly on shape."
  },
  {
    id: "svgcards-vertical4",
    title: "SVGCards Vertical4",
    source: "svgcards",
    sourceLabel: "saulspatz/SVGCards Decks/Vertical4",
    license: "Public Domain",
    npm: "not needed for this fixture",
    vendor: "11 PNG fixtures",
    assetFolder: "vertical4",
    aspectRatio: svgCardsPngAspectRatio,
    summary: "Traditional stacked index with four-color suits; good large-card readability."
  },
  {
    id: "svgcards-horizontal4",
    title: "SVGCards Horizontal4",
    source: "svgcards",
    sourceLabel: "saulspatz/SVGCards Decks/Horizontal4",
    license: "Public Domain",
    npm: "not needed for this fixture",
    vendor: "11 PNG fixtures",
    assetFolder: "horizontal4",
    aspectRatio: svgCardsPngAspectRatio,
    summary: "Side-by-side rank and suit index; strongest SVGCards option for river 25% clipping."
  },
  {
    id: "svgcards-accessible-horizontal",
    title: "SVGCards Accessible",
    source: "svgcards",
    sourceLabel: "saulspatz/SVGCards Decks/Accessible/Horizontal",
    license: "Public Domain",
    npm: "not needed for this fixture",
    vendor: "11 PNG fixtures",
    assetFolder: "accessible-horizontal",
    aspectRatio: svgCardsPngAspectRatio,
    summary: "Higher-contrast four-color palette; river clipping remains readable."
  },
  {
    id: "cardmeister",
    title: "cardmeister 4-color",
    source: "cardmeister",
    sourceLabel: "cardmeister/cardmeister.github.io",
    license: "Unlicense",
    npm: "no npm package found",
    vendor: "elements.cardmeister.full.js",
    aspectRatio: leteleAspectRatio,
    selected: true,
    summary: "Configurable web component using suitcolor/rankcolor; easy palette iteration, but custom element integration is separate from React."
  }
] as const satisfies readonly CardDesignCandidate[];

const comparisonModes = [
  {
    title: "CurrentTrick",
    metric: `${currentTrickEquivalent.width} x ${currentTrickEquivalent.height}`,
    className: "current",
    cards: currentTrickComparisonCards,
    displayWidth: 82,
    sourceWidth: currentTrickEquivalent.width
  },
  {
    title: "Self hand",
    metric: `${selfHandEquivalent.width} x ${selfHandEquivalent.height}`,
    className: "self",
    cards: fourColorComparisonCards,
    displayWidth: 42,
    sourceWidth: selfHandEquivalent.width
  },
  {
    title: "River 25%",
    metric: `${riverEquivalent.width}w / ${riverEquivalent.visibleHeight}h`,
    className: "river",
    cards: tenExposureCards,
    displayWidth: 64,
    sourceWidth: riverEquivalent.width,
    clipped: true
  }
] as const satisfies readonly CandidateComparisonMode[];

export function CardDesignMock() {
  const normalWidth = cardDesignConfig.sizes.normalWidth;
  const smallWidth = cardDesignConfig.sizes.smallWidth;
  const overlapWidth = cardDesignConfig.sizes.overlapWidth;
  const identificationGuideX = cardDesignIdentificationGuideX(overlapWidth);
  const exposureOffset = cardDesignExposureOffset(overlapWidth);

  useCardmeisterScript();

  return (
    <main aria-label="Issue 392 card design mock" className="card-design-page">
      <header className="card-design-header">
        <div>
          <p className="card-design-eyebrow">/mock/card-design</p>
          <h1>Four-Color Card Design Sandbox</h1>
        </div>
        <ConfigSwatches />
      </header>

      <section aria-label="4色カード候補比較" className="card-design-section card-design-section-wide card-design-candidate-section">
        <SectionTitle metric={`${cardDesignCandidates.length} candidates / 11 cards`} title="Four-color candidates" />
        <div className="card-design-candidate-grid">
          {cardDesignCandidates.map((candidate) => (
            <CandidatePanel candidate={candidate} key={candidate.id} />
          ))}
        </div>
      </section>

      <section aria-label="通常サイズ" className="card-design-section card-design-section-primary">
        <SectionTitle metric={`${normalWidth} x ${cardDesignCardHeight(normalWidth).toFixed(1)}px`} title="Normal" />
        <div className="card-design-focus-grid">
          {cardDesignSuitOrder.map((suit) => (
            <SuitRow key={suit} suit={suit} width={normalWidth} />
          ))}
        </div>
      </section>

      <section aria-label="小サイズ" className="card-design-section">
        <SectionTitle metric={`${smallWidth} x ${cardDesignCardHeight(smallWidth).toFixed(1)}px`} title="Small" />
        <div className="card-design-small-grid">
          {focusCards.map((card) => (
            <CardDesignPrototypeCard card={card} className="card-design-card-small" key={`${card.rank}-${card.suit}`} width={smallWidth} />
          ))}
        </div>
      </section>

      <section aria-label="25%露出テスト" className="card-design-section">
        <SectionTitle
          metric={`guide ${Math.round(cardDesignConfig.layout.identificationAreaRatio * 100)}% / offset ${Math.round(cardDesignConfig.layout.exposureOffsetRatio * 100)}%`}
          title="25% Exposure"
        />
        <div
          className="card-design-exposure-debug"
          style={
            {
              "--card-design-identification-guide-x": `${identificationGuideX}px`,
              "--card-design-overlap-card-width": `${overlapWidth}px`,
              "--card-design-overlap-card-height": `${cardDesignCardHeight(overlapWidth)}px`,
              "--card-design-overlap-count": tenExposureCards.length,
              "--card-design-overlap-step": `${cardDesignConfig.layout.exposureOffsetRatio * 100}%`,
              "--card-design-exposure-offset": `${exposureOffset}px`,
              "--card-design-overlap-width": `${cardDesignOverlapWidth(tenExposureCards.length, overlapWidth)}px`
            } as CSSProperties
          }
        >
          <div aria-hidden="true" className="card-design-exposure-ruler">
            <span className="card-design-exposure-ruler-segment card-design-exposure-ruler-segment-left">25%</span>
            <span className="card-design-exposure-ruler-segment card-design-exposure-ruler-segment-right">75%</span>
          </div>
          <div className="card-design-overlap-strip">
            <div aria-hidden="true" className="card-design-overlap-guides">
              {tenExposureCards.slice(0, -1).map((card, index) => (
                <span
                  className="card-design-overlap-guide"
                  key={`${card.rank}-${card.suit}-guide`}
                  style={
                    {
                      "--card-design-overlap-guide-left": `${cardDesignExposureLeft(index + 1, overlapWidth)}px`
                    } as CSSProperties
                  }
                />
              ))}
            </div>
            {tenExposureCards.map((card, index) => (
              <div
                className="card-design-overlap-card"
                key={`${card.rank}-${card.suit}`}
                style={
                  {
                    "--card-design-overlap-left": `${cardDesignExposureLeft(index, overlapWidth)}px`,
                    zIndex: index + 1
                  } as CSSProperties
                }
              >
                <CardDesignPrototypeCard card={card} showIdentificationGuide width={overlapWidth} />
              </div>
            ))}
          </div>
        </div>
      </section>

      <section aria-label="JQK比較" className="card-design-section">
        <SectionTitle metric="rank + suit" title="J / Q / K" />
        <div className="card-design-face-grid">
          {faceComparisonCards.map((card) => (
            <CardDesignPrototypeCard card={card} key={`${card.rank}-${card.suit}`} width={86} />
          ))}
        </div>
      </section>

      <section aria-label="52枚一覧" className="card-design-section card-design-section-wide">
        <SectionTitle metric="52 cards" title="Full Deck" />
        <div className="card-design-deck-grid">
          {fullDeckCards.map((card) => (
            <CardDesignPrototypeCard card={card} className="card-design-card-deck" key={`${card.rank}-${card.suit}`} width={48} />
          ))}
        </div>
      </section>

      <section aria-label="@letele/playing-cards比較" className="card-design-section card-design-section-wide">
        <SectionTitle metric="current" title="@letele comparison" />
        <div className="card-design-letele-grid">
          {fourColorComparisonCards.map((card) => (
            <LeteleCard card={card} key={`${card.rank}-${card.suit}`} />
          ))}
        </div>
      </section>
    </main>
  );
}

function CandidatePanel({ candidate }: { candidate: CardDesignCandidate }) {
  return (
    <article aria-label={`${candidate.title} comparison`} className="card-design-candidate-card">
      <div className="card-design-candidate-heading">
        <h3>
          {candidate.title}
          {candidate.selected === true ? <span className="card-design-selected-label">Selected</span> : null}
        </h3>
        <span>{candidate.license}</span>
      </div>
      <dl className="card-design-candidate-meta">
        <div>
          <dt>source</dt>
          <dd>{candidate.sourceLabel}</dd>
        </div>
        <div>
          <dt>npm</dt>
          <dd>{candidate.npm}</dd>
        </div>
        <div>
          <dt>vendor</dt>
          <dd>{candidate.vendor}</dd>
        </div>
      </dl>
      {comparisonModes.map((mode) => (
        <CandidateModeRow candidate={candidate} key={mode.title} mode={mode} />
      ))}
      <p className="card-design-candidate-summary">{candidate.summary}</p>
    </article>
  );
}

function CandidateModeRow({
  candidate,
  mode
}: {
  candidate: CardDesignCandidate;
  mode: CandidateComparisonMode;
}) {
  return (
    <div className={`card-design-candidate-mode card-design-candidate-mode-${mode.className}`}>
      <div className="card-design-candidate-mode-title">
        <strong>{mode.title}</strong>
        <span>{mode.metric}</span>
      </div>
      <div className="card-design-candidate-strip">
        {mode.cards.map((card) => (
          <CandidateCard
            card={card}
            candidate={candidate}
            clipped={mode.clipped}
            key={`${candidate.id}-${mode.title}-${card.rank}-${card.suit}`}
            sourceWidth={mode.sourceWidth}
            width={mode.displayWidth}
          />
        ))}
      </div>
    </div>
  );
}

function CandidateCard({
  candidate,
  card,
  clipped = false,
  sourceWidth,
  width
}: {
  candidate: CardDesignCandidate;
  card: MockPlayingCard;
  clipped?: boolean;
  sourceWidth: number;
  width: number;
}) {
  const height = width * candidate.aspectRatio;
  const visibleHeight = clipped ? height * cardDesignConfig.layout.identificationAreaRatio : height;

  return (
    <div
      aria-label={`${candidate.title} ${card.rank}${cardDesignSuitSymbols[card.suit]}`}
      className={[
        "card-design-candidate-card-frame",
        clipped ? "card-design-candidate-card-frame-clipped" : ""
      ].filter(Boolean).join(" ")}
      style={
        {
          "--card-design-candidate-card-height": `${height}px`,
          "--card-design-candidate-card-source-width": `${sourceWidth}px`,
          "--card-design-candidate-card-visible-height": `${visibleHeight}px`,
          "--card-design-candidate-card-width": `${width}px`
        } as CSSProperties
      }
    >
      <div className="card-design-candidate-card-clip">
        {candidate.source === "letele" ? <LeteleCard card={card} /> : null}
        {candidate.source === "svgcards" ? <SvgCardsImageCard assetFolder={candidate.assetFolder ?? ""} card={card} /> : null}
        {candidate.source === "cardmeister" ? <CardmeisterCard card={card} /> : null}
      </div>
    </div>
  );
}

function SvgCardsImageCard({ assetFolder, card }: { assetFolder: string; card: MockPlayingCard }) {
  return (
    <img
      alt=""
      className="card-design-candidate-image"
      src={resolveAppPath(`/vendor/card-design/svgcards/${assetFolder}/${svgCardsFileName(card)}`)}
    />
  );
}

function CardmeisterCard({ card }: { card: MockPlayingCard }) {
  return <CardmeisterPlayingCard card={card} className="card-design-cardmeister-element" />;
}

function SectionTitle({ metric, title }: { metric: string; title: string }) {
  return (
    <div className="card-design-section-title">
      <h2>{title}</h2>
      <span>{metric}</span>
    </div>
  );
}

function SuitRow({ suit, width }: { suit: MockPlayingCardSuit; width: number }) {
  return (
    <div aria-label={`${cardDesignSuitLabels[suit]} comparison row`} className="card-design-suit-row">
      <div
        className="card-design-suit-label"
        style={
          {
            "--card-design-suit-color": cardDesignConfig.colors[suit]
          } as CSSProperties
        }
      >
        <strong>{cardDesignSuitSymbols[suit]}</strong>
        <span>{cardDesignSuitLabels[suit]}</span>
      </div>
      <div className="card-design-row-cards">
        {cardDesignComparisonRanks.map((rank) => (
          <CardDesignPrototypeCard card={{ rank, suit }} key={`${rank}-${suit}`} showIdentificationGuide width={width} />
        ))}
      </div>
    </div>
  );
}

function ConfigSwatches() {
  return (
    <div aria-label="スートカラー設定" className="card-design-config-swatches">
      {cardDesignSuitOrder.map((suit) => (
        <span
          className="card-design-config-swatch"
          key={suit}
          style={
            {
              "--card-design-suit-color": cardDesignConfig.colors[suit]
            } as CSSProperties
          }
        >
          {cardDesignSuitSymbols[suit]} {cardDesignConfig.colors[suit]}
        </span>
      ))}
    </div>
  );
}

function LeteleCard({ card }: { card: MockPlayingCard }) {
  const CardComponent = mockPlayingCardComponent(card);
  const componentName = mockPlayingCardComponentName(card);

  return (
    <article aria-label={`@letele ${componentName}`} className="card-design-letele-card">
      <CardComponent className="card-design-letele-svg" title={componentName} />
    </article>
  );
}

function svgCardsFileName(card: MockPlayingCard): string {
  const rankNames: Partial<Record<MockPlayingCard["rank"], string>> = {
    A: "Ace",
    J: "Jack",
    Q: "Queen",
    K: "King"
  };
  const rankName = rankNames[card.rank] ?? card.rank;
  const suitName = {
    spades: "spade",
    clubs: "club",
    hearts: "heart",
    diamonds: "diamond"
  }[card.suit];

  return `${suitName}${rankName}.png`;
}
