export type ClassName = 'Low' | 'Medium' | 'High';

export interface ModelRow {
  model: string;
  display: string;
  in_paper: boolean;
  accuracy: number;
  accuracy_std?: number;
  f1_weighted: number;
  f1_weighted_std?: number;
  f1_macro: number;
  f1_val?: number;
  f1_val_std?: number;
  fit_seconds: number;
  confusion: number[][];
  f1_per_class: number[];
}

export interface Results {
  generated_at: string;
  paper: {
    citation: string;
    split: string;
    buildings: number;
    table3_performance_class: Record<string, { f1_val: number; f1_test: number; acc_val: number; acc_test: number }>;
    fig6a_f1_performance_class: Record<string, number>;
  };
  cohort: {
    buildings: number;
    sites: number;
    class_counts: Record<string, number>;
    class_cutoffs_kwh_m2_yr: number[];
    label_note: string;
  };
  dataset_100k: {
    rows: number;
    columns: number;
    feature_columns: number;
    buildings: number;
    sites: number;
    unit: string;
  };
  paper_protocol: ModelRow[];
  scale_protocol: { random_rows: ModelRow[]; whole_buildings: ModelRow[] };
  best_model_whole_buildings: string;
  served_model: string;
  feature_importance_top15: [string, number][];
  classes: ClassName[];
}

export interface Example {
  building_id: string;
  site_id: string;
  primaryspaceusage: string;
  week_start: string;
  true_class: ClassName;
  hourly_kwh: (number | null)[];
  air_temperature: number[];
}

export interface Prediction {
  prediction: ClassName;
  probabilities: Record<ClassName, number>;
  model: string;
}

export interface Health {
  status: string;
  inference_mode: string;
  model: string | null;
  feature_count: number;
}

export interface ProtocolRow {
  protocol: string;
  label: string;
  description: string;
  models: {
    model: string;
    display: string;
    accuracy: number;
    f1_weighted: number;
    f1_macro: number;
    building_f1_weighted?: number;
    building_f1_weighted_std?: number;
    ci_low?: number;
    ci_high?: number;
  }[];
}

export interface Novelty {
  generated_at: string;
  headline: string;
  protocols: ProtocolRow[];
  per_site?: {
    site: string;
    buildings: number;
    accuracy: number;
    f1_weighted: number;
    use_type_accuracy?: number;
    use_type_f1_weighted?: number;
  }[];
  notes?: string[];
}
