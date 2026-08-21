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
  mockPlayingCardComponent,
  mockPlayingCardComponentName,
  type MockPlayingCard,
  type MockPlayingCardSuit
} from "./mockPlayingCardAdapter";
import "./CardDesignMock.css";

const focusCards = createCardDesignDeck(cardDesignComparisonRanks);
const fullDeckCards = createCardDesignDeck();
const tenExposureCards = cardDesignSuitOrder.map((suit) => ({ rank: "10", suit })) satisfies readonly MockPlayingCard[];
const faceComparisonCards = createCardDesignDeck(["J", "Q", "K"]);
const leteleComparisonCards: readonly MockPlayingCard[] = [
  { rank: "A", suit: "spades" },
  { rank: "A", suit: "clubs" },
  { rank: "A", suit: "hearts" },
  { rank: "A", suit: "diamonds" },
  { rank: "10", suit: "spades" },
  { rank: "10", suit: "clubs" },
  { rank: "10", suit: "hearts" },
  { rank: "10", suit: "diamonds" }
];

export function CardDesignMock() {
  const normalWidth = cardDesignConfig.sizes.normalWidth;
  const smallWidth = cardDesignConfig.sizes.smallWidth;
  const overlapWidth = cardDesignConfig.sizes.overlapWidth;
  const identificationGuideX = cardDesignIdentificationGuideX(overlapWidth);
  const exposureOffset = cardDesignExposureOffset(overlapWidth);

  return (
    <main aria-label="Issue 355 card design mock" className="card-design-page">
      <header className="card-design-header">
        <div>
          <p className="card-design-eyebrow">/mock/card-design</p>
          <h1>Card Design Sandbox</h1>
        </div>
        <ConfigSwatches />
      </header>

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
          {leteleComparisonCards.map((card) => (
            <LeteleCard card={card} key={`${card.rank}-${card.suit}`} />
          ))}
        </div>
      </section>
    </main>
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
