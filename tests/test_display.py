"""0.28: a game window on another monitor is moved onto the primary display, which is the only one
DXGI capture reads [runs 20261029-045632, -045723, -124858: 'capture region (-2559, 0, 1278, 1399)
is outside the primary output (2560x1440)'; user review of run 20261029-124921]."""

from shogunbot.display import move_to, on_primary

PRIMARY = (2560, 1440)


def test_a_window_on_the_primary_display_stays():
    assert on_primary((0, 0, 2560, 1440), PRIMARY)
    assert move_to((100, 31, 1920, 1080), (92, 0, 2028, 1119), PRIMARY) is None


def test_the_window_of_run_124858_goes_to_the_origin():
    # client (x, y, w, h) as winio.rect() gives it; the window rect with an 8 px border and a title bar
    client = (-2559, 0, 1278, 1399)
    outer = (-2567, -31, -1273, 1407)
    assert not on_primary(client, PRIMARY)
    assert move_to(client, outer, PRIMARY) == (0, 0)


def test_a_window_too_big_for_the_display_keeps_its_client_area_on_it():
    # borders off screen: the client area starts at the origin
    client = (-2560, 0, 2560, 1440)
    outer = (-2568, -31, 8, 1448)
    x, y = move_to(client, outer, PRIMARY)
    assert (x, y) == (-8, -31)
    assert on_primary((x + client[0] - outer[0], y + client[1] - outer[1], 2560, 1440), PRIMARY)
