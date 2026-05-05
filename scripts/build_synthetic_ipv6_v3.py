
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.synthetic_ipv6_generator import SyntheticIPv6GroundedV3Generator, zip_bundle


def main():
    ap = argparse.ArgumentParser(description='Build a grounded synthetic IPv6 v3-style bundle with 32x32 thumbnails.')
    ap.add_argument('--reference-zip', required=True, help='Path to the existing synthetic_ipv6_grounded_v3_32x32.zip reference bundle.')
    ap.add_argument('--output-root', required=True, help='Directory where the regenerated bundle will be written.')
    ap.add_argument('--profile', default='from_reference', choices=['from_reference', 'balanced_bootstrap'], help='Generation profile.')
    ap.add_argument('--image-size', type=int, default=32, help='Thumbnail size. Keep at 32 for v3 compatibility.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-total', type=int, default=None, help='Optional total number of rows for rescaled generation.')
    ap.add_argument('--zip-output', action='store_true', help='Also create a zip package after generation.')
    args = ap.parse_args()

    gen = SyntheticIPv6GroundedV3Generator(reference_zip=args.reference_zip, image_size=args.image_size, seed=args.seed)
    bundle = gen.generate(output_root=args.output_root, profile=args.profile, n_total=args.n_total)
    bundle_root = Path(args.output_root) / f'synthetic_ipv6_grounded_v3_{args.image_size}x{args.image_size}'
    print(f'Generated bundle at: {bundle_root}')
    if args.zip_output:
        out_zip = zip_bundle(bundle_root)
        print(f'Zip package written to: {out_zip}')


if __name__ == '__main__':
    main()
