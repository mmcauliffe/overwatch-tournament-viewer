import os

txt = '''2412
2413
2414
2415
2433
2434
2435
2436
2437
2438
2439
2441
2442
2443
2444
2445
2446
2447
2448
2450
2451
2452
2453
2454
2456
2457
2458
2459
2460
2461
2462
2463
2464
2465
2466
2467
2468
2470
2471
2472
3887
3888
3889'''

rounds = [int(x) for x in txt.split('\n') if x]

spec_modes = [
    'original',
    'overwatch league',
    'world cup',
    'contenders']

train_dir = r'N:\Data\Overwatch\training_data\game'

for m in spec_modes:
    m_dir = os.path.join(train_dir, m)
    if os.path.exists(m_dir):
        for f in os.listdir(m_dir):
            if not f.endswith('.hdf5'):
                continue
            round_id = int(os.path.splitext(f)[0])
            if round_id in rounds:
                print('deleting {}'.format(round_id))
                os.remove(os.path.join(m_dir, f))