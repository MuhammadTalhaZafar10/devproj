import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { PredictionApiService } from './prediction-api.service';
import { TracePanelComponent } from './trace-panel.component';
import { FeatureFormComponent } from './feature-form.component';
import { Example, Health, ModelRow, Novelty, Results } from './models';

const PLOT = { w: 640, left: 150, right: 24, min: 0.45, max: 0.75 };

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [DecimalPipe, TracePanelComponent, FeatureFormComponent],
  templateUrl: './app.component.html',
})
export class AppComponent implements OnInit {
  private readonly api = inject(PredictionApiService);

  readonly results = signal<Results | null>(null);
  readonly examples = signal<Example[]>([]);
  readonly novelty = signal<Novelty | null>(null);
  readonly health = signal<Health | null>(null);
  readonly loadError = signal(false);
  readonly plot = PLOT;

  readonly paperTest = computed(() => this.results()?.paper.table3_performance_class ?? null);

  readonly rows = computed<ModelRow[]>(() => this.results()?.paper_protocol ?? []);

  /** Paper figure 6a gives CV F1 for RF and GB, the two models we share with it. */
  paperCv(model: string): number | null {
    return this.results()?.paper.fig6a_f1_performance_class[model] ?? null;
  }

  readonly gap = computed(() => {
    const r = this.results();
    if (!r) return [];
    return r.scale_protocol.random_rows.map((row) => {
      const whole = r.scale_protocol.whole_buildings.find((w) => w.model === row.model)!;
      return { display: row.display, random: row.f1_weighted, whole: whole.f1_weighted };
    });
  });

  readonly best = computed(() => {
    const r = this.results();
    if (!r) return null;
    return r.scale_protocol.whole_buildings.find((m) => m.model === r.best_model_whole_buildings) ?? null;
  });

  /** Accuracy of the served model on single weeks from whole buildings it has not seen. */
  readonly weekAccuracy = computed(() => {
    const r = this.results();
    if (!r) return null;
    return r.scale_protocol.whole_buildings.find((m) => m.model === r.served_model)?.accuracy ?? null;
  });

  readonly confusion = computed(() => {
    const b = this.best();
    if (!b) return [];
    return b.confusion.map((row) => {
      const total = row.reduce((a, c) => a + c, 0) || 1;
      return row.map((v) => ({ n: v, share: v / total }));
    });
  });

  readonly importance = computed(() => {
    const r = this.results();
    if (!r) return [];
    const top = r.feature_importance_top15;
    const max = Math.max(...top.map((t) => t[1]), 1);
    return top.map(([name, v]) => ({ name: this.pretty(name), share: v / max, v }));
  });

  ngOnInit(): void {
    this.api.results().subscribe({
      next: (r) => this.results.set(r),
      error: () => this.loadError.set(true),
    });
    this.api.examples().subscribe({ next: (e) => this.examples.set(e), error: () => undefined });
    this.api.novelty().subscribe((n) => this.novelty.set(n));
    this.api.health().subscribe({ next: (h) => this.health.set(h), error: () => this.health.set(null) });
  }

  x(v: number): number {
    const span = PLOT.w - PLOT.left - PLOT.right;
    return PLOT.left + ((Math.min(Math.max(v, PLOT.min), PLOT.max) - PLOT.min) / (PLOT.max - PLOT.min)) * span;
  }

  ticks(): number[] {
    return [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75];
  }

  pct(v: number): string {
    return `${(v * 100).toFixed(1)}%`;
  }

  pretty(name: string): string {
    return name
      .replace(/^hour_(\d+)/, 'hour $1:00')
      .replace(/^dow_(\d)/, (_, d) => ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][+d] + ' level')
      .replace(/^q(\d+)$/, 'percentile $1')
      .replace(/_/g, ' ');
  }

  heat(share: number): string {
    return `color-mix(in srgb, var(--teal) ${Math.round(share * 85)}%, var(--panel))`;
  }

  className(i: number): string {
    return ['Low', 'Medium', 'High'][i];
  }
}
