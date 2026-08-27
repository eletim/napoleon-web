import type { CSSProperties } from "react";
import { cardDesignSuitLabels, cardDesignSuitOrder, cardDesignSuitSymbols } from "./CardDesignCard";
import {
  CardmeisterPlayingCard,
  useCardmeisterScript
} from "./CardmeisterPlayingCard";
import { cardmeisterFourColorCsv, fourColorSuitColors } from "./cardSuitTheme";
import type { MockPlayingCard } from "./mockPlayingCardAdapter";
import "./CardDesignMock.css";

interface CardPreviewGroup {
  cards: readonly MockPlayingCard[];
  className: string;
  metric: string;
  title: string;
  width: number;
}

const selectedCards: readonly MockPlayingCard[] = [
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

const currentTrickCards = selectedCards.slice(8);
const selfHandCards = selectedCards;
const cardAspectRatio = 7 / 5;

const previewGroups = [
  {
    title: "CurrentTrick相当",
    metric: "342.939 x 480.115 world px reference",
    className: "current",
    cards: currentTrickCards,
    width: 112
  },
  {
    title: "自分手札相当",
    metric: "118.154 x 165.415 world px reference",
    className: "self",
    cards: selfHandCards,
    width: 58
  }
] as const satisfies readonly CardPreviewGroup[];

export function CardDesignMock() {
  useCardmeisterScript();

  return (
    <main aria-label="cardmeister selected card design mock" className="card-design-page">
      <header className="card-design-header">
        <div>
          <p className="card-design-eyebrow">/mock/card-design</p>
          <h1>cardmeister 4-color selected</h1>
        </div>
        <ConfigSwatches />
      </header>

      <section aria-label="採用カード概要" className="card-design-section card-design-section-summary">
        <div>
          <p className="card-design-selected-label">Selected</p>
          <h2>cardmeister 4-color</h2>
          <p>
            Production and retained table mocks use the cardmeister custom element for normal face-up cards.
            The shared suit theme is applied through both <code>suitcolor</code> and <code>rankcolor</code>.
          </p>
        </div>
        <dl className="card-design-meta">
          <div>
            <dt>source</dt>
            <dd>cardmeister/cardmeister.github.io</dd>
          </div>
          <div>
            <dt>license</dt>
            <dd>Unlicense</dd>
          </div>
          <div>
            <dt>npm</dt>
            <dd>no npm package used</dd>
          </div>
          <div>
            <dt>vendor</dt>
            <dd>/vendor/card-design/cardmeister/elements.cardmeister.full.js</dd>
          </div>
          <div>
            <dt>colors</dt>
            <dd>{cardmeisterFourColorCsv}</dd>
          </div>
        </dl>
      </section>

      <section aria-label="採用カード確認" className="card-design-section card-design-section-wide">
        <SectionTitle metric={`${selectedCards.length} cards`} title="Selected card check" />
        <div className="card-design-selected-grid">
          {selectedCards.map((card) => (
            <CardFrame card={card} key={`${card.rank}-${card.suit}`} width={88} />
          ))}
        </div>
      </section>

      <section aria-label="サイズ別確認" className="card-design-section card-design-section-wide">
        <SectionTitle metric="normal face-up card only" title="Production size references" />
        <div className="card-design-preview-groups">
          {previewGroups.map((group) => (
            <article className={`card-design-preview-group card-design-preview-group-${group.className}`} key={group.title}>
              <div className="card-design-preview-heading">
                <h3>{group.title}</h3>
                <span>{group.metric}</span>
              </div>
              <div className="card-design-preview-strip">
                {group.cards.map((card) => (
                  <CardFrame card={card} key={`${group.title}-${card.rank}-${card.suit}`} width={group.width} />
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}

function CardFrame({ card, width }: { card: MockPlayingCard; width: number }) {
  return (
    <article
      aria-label={`${card.rank}${cardDesignSuitSymbols[card.suit]}`}
      className="card-design-card-frame"
      style={
        {
          "--card-design-card-height": `${width * cardAspectRatio}px`,
          "--card-design-card-width": `${width}px`
        } as CSSProperties
      }
    >
      <CardmeisterPlayingCard card={card} className="card-design-cardmeister-element" />
    </article>
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
              "--card-design-suit-color": fourColorSuitColors[suit]
            } as CSSProperties
          }
        >
          {cardDesignSuitSymbols[suit]} {cardDesignSuitLabels[suit]} {fourColorSuitColors[suit]}
        </span>
      ))}
    </div>
  );
}

function SectionTitle({ metric, title }: { metric: string; title: string }) {
  return (
    <div className="card-design-section-title">
      <h2>{title}</h2>
      <span>{metric}</span>
    </div>
  );
}
