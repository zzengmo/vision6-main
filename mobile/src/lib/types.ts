export type Photo = { uri: string; name: string; mime: string; width: number; height: number; size?: number };
export type Policy = { id: string; label: string; threshold: number };
export type Model = { id: string; label: string; ready: boolean; input: string; policies: Policy[] };
export type Metadata = {
  models: Model[];
  revision: string;
  segmentation: null | { epochs: number; threshold: number; test: { dice: number; iou: number } };
  limits: { bytes: number; pixels: number; unet_size: [number, number] };
  examples: { filename: string; class_id: number; available: boolean }[];
};
export type Classification = {
  model_id: string; decision: 'normal' | 'defect'; score: number; threshold: number;
  elapsed_seconds: number; original_url: string; csv_url: string;
};
export type Box = { box: number; x: number; y: number; width: number; height: number; mean_score: number };
export type Localization = {
  boxes: Box[]; mask_pixels: number; elapsed_seconds: number; threshold: number;
  original_url: string; overlay_url: string; mask_url: string; candidates_url: string; csv_url: string;
};
