import sys
import spiceypy as spice

if len(sys.argv) != 2:
    print("Usage: python inspect_spk_kernel.py /path/to/orbit.bsp")
    raise SystemExit(1)

spk = sys.argv[1]
spice.furnsh("/Users/dedo/Downloads/naif0012.tls")
spice.furnsh(spk)

try:
    object_ids = sorted(spice.spkobj(spk))

    if not object_ids:
        print("No object IDs found in the SPK.")
        raise SystemExit(0)

    for object_id in object_ids:
        print(f"\nObject ID: {object_id}")
        coverage = spice.spkcov(spk, object_id)

        for i in range(spice.wncard(coverage)):
            start_et, end_et = spice.wnfetd(coverage, i)

            print(f"  Coverage start ET:  {start_et:.3f}")
            print(f"  Coverage end ET:    {end_et:.3f}")
            print(f"  Coverage start UTC: {spice.et2utc(start_et, 'C', 3)}")
            print(f"  Coverage end UTC:   {spice.et2utc(end_et, 'C', 3)}")

            midpoint = 0.5 * (start_et + end_et)
            handle, descriptor, found = spice.spksfs(object_id, midpoint, 256)

            if found:
                target, center, frame, data_type, first, last, begin, end = (
                    spice.spkuds(descriptor)
                )
                print(f"  Segment target: {target}")
                print(f"  Segment center: {center}")
                print(f"  Segment frame ID: {frame}")
                print(f"  Segment data type: {data_type}")
finally:
    spice.kclear()
