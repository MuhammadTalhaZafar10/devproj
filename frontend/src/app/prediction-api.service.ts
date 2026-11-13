import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class PredictionApiService {
  constructor(private readonly http: HttpClient) {}

  health() {
    return this.http.get<{
      status: string;
      inference_mode: string;
      feature_count: number;
      model_version: string;
    }>('/api/health');
  }

  predict(payload: Record<string, number>) {
    return this.http.post<{
      prediction: string;
      inference_mode: string;
      model_version: string;
      probabilities: Record<string, number>;
      conformal_prediction_set: string[];
      note?: string;
    }>('/api/predict', payload);
  }
}
