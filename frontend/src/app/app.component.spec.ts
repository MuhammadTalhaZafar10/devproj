import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { AppComponent } from './app.component';
import { Results } from './models';

const row = (model: string, display: string, inPaper: boolean, f1: number) => ({
  model, display, in_paper: inPaper, accuracy: f1, f1_weighted: f1, f1_weighted_std: 0.01, f1_macro: f1,
  f1_val: f1, fit_seconds: 1, confusion: [[8, 1, 1], [1, 8, 1], [1, 1, 8]], f1_per_class: [f1, f1, f1],
});

const results: Results = {
  generated_at: '2026-11-27T00:00:00Z',
  paper: {
    citation: 'Raymand et al. (2023)', split: '20 % test', buildings: 1494,
    table3_performance_class: { random_forest_all_features: { f1_val: 0.6, f1_test: 0.604, acc_val: 0.6, acc_test: 0.605 },
      optimal_pipeline: { f1_val: 0.62, f1_test: 0.622, acc_val: 0.62, acc_test: 0.622 } },
    fig6a_f1_performance_class: { random_forest: 0.62, gradient_boosting: 0.6 },
  },
  cohort: { buildings: 1470, sites: 18, class_counts: { Low: 490 }, class_cutoffs_kwh_m2_yr: [78, 158], label_note: '' },
  dataset_100k: { rows: 100000, columns: 59, feature_columns: 54, buildings: 1468, sites: 18, unit: 'week' },
  paper_protocol: [row('random_forest', 'Random Forest', true, 0.6), row('lightgbm', 'LightGBM', false, 0.62)],
  scale_protocol: {
    random_rows: [row('random_forest', 'Random Forest', true, 0.95), row('lightgbm', 'LightGBM', false, 0.97)],
    whole_buildings: [row('random_forest', 'Random Forest', true, 0.72), row('lightgbm', 'LightGBM', false, 0.73)],
  },
  best_model_whole_buildings: 'lightgbm',
  served_model: 'lightgbm',
  feature_importance_top15: [['hour_03', 10], ['cv', 5]],
  classes: ['Low', 'Medium', 'High'],
};

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AppComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
  });

  it('creates the app', () => {
    expect(TestBed.createComponent(AppComponent).componentInstance).toBeTruthy();
  });

  it('renders the comparison table from the results file', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const http = TestBed.inject(HttpTestingController);
    fixture.detectChanges();
    http.expectOne('classification_results.json').flush(results);
    http.match(() => true).forEach((r) => r.flush(null, { status: 404, statusText: 'x' }));
    fixture.detectChanges();
    const text: string = fixture.nativeElement.textContent;
    expect(text).toContain('Random Forest');
    expect(text).toContain('Added by us');
    expect(text).toContain('100,000');
  });
});
