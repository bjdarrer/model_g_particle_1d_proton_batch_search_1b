Date: 9 March 2026 17:56 GMT

Model G 1D proton-like batch search (non-GUI)

............
- Written by Brendan Darrer aided by ChatGPT 5.4 (extensive thinking)
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
  
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.

Install:

    pip3 install numpy scipy matplotlib imageio imageio[ffmpeg]

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+

............

Built from the same 1D Model G equations used in Brendan's GUI explorer,
but refactored into a command-line batch search that scores each trial
against a Kelly-style proton target profile.

What it does

------------
1) Builds a Kelly-style proton target from the electric Sachs form factor G_E^p(Q^2)
   using a Kelly-style relativistic inversion to rho(r).
2) Runs many 1D Model G simulations over a user-defined parameter grid.
3) Compares the final pY(x) profile against the Kelly target by shape only
   (normalized profile + mirrored surface profile x^2 rho), because in SQK the
   substrate-potential units are being used as a proxy rather than literal
   charge-density units.
4) Writes CSV rankings and a PNG comparing the best SQK candidate with the target.

Typical use:

    python3 model_g_particle_1d_proton_batch_search_1b.py

Faster smoke test

    python3 model_g_particle_1d_proton_batch_search_1b.py --nx 41 --tfinal 6 --dy 9.5,10.5,12 --b 27,28,29

Note to Brendan (not necessarily others):

    python3 -m venv tf-env

    source tf-env/bin/activate

..............

"The best next step is for me to turn your GUI code into a non-GUI batch search script that automatically

scores each run against a Kelly-style proton target profile and writes out the top candidate parameter 

sets." Yes, thanks. Can you do this?

ChatGPT 5.4:

Thought for 26m 49s

Yes — I made the non-GUI batch search script and also ran a quick coarse search.

Files:

- Batch search script
- Quick-search summary
- Quick-search top candidates CSV
- Best candidate vs Kelly proton PNG
- Best candidate profiles NPZ

What the script does:

uses the same 1D Model G equations as your GUI

builds a Kelly-style proton target profile

compares the final SQK pY(x) shape against that target after normalization

also compares the mirrored surface profile 

x^2 pY(x)

writes ranked candidate parameter sets to CSV

From the quick coarse run I did, the current best provisional proton-like candidate was:

dy = 12.0

b = 29.0

g = 0.09

amp = 1.0

sx = 1.0

st = 1.5

Tseed = 3.0

sign = -1

with:

pY_core = 0.288756...

Qproxy_int_pYdx = 0.164173...

polarity = positive-polarity (proton-like)

One caveat: that run was intentionally coarse (nx=41, Tfinal=6) just to get a fast first pass, so this is a provisional best candidate, not yet a high-resolution final fit.

You can run the script like this:

python3 model_g_particle_1d_proton_batch_search_1b.py

or for the same quick sweep I used:

python3 model_g_particle_1d_proton_batch_search_1b.py \
  --outdir /mnt/data/model_g_proton_batch_search_quick_1b \
  --nx 41 --tfinal 6 \
  --dy 9.5,10.5,12.0 \
  --b 27,28,29 \
  --g 0.09,0.10,0.11 \
  --amp 1.0 \
  --sx 0.9,1.0,1.1 \
  --st 1.5 \
  --Tseed 3.0

The natural next step is a second-stage refinement around that neighbourhood with a finer grid and longer runtime.

Brendan: Thank you ChatGPT! I'll be very interested to review work you've just done.

ChatGPT 5.4:

Thought for 5s

You’re very welcome, Brendan.

A good way to review it is to start with the plot and summary first:

Best candidate vs Kelly proton PNG

Quick-search summary

Top candidates CSV

Then, when you want the mechanics, look at:

Batch search script

Best candidate profiles NPZ

The main thing to check is whether the best candidate really gives you the kind of single broad positive proton-like core you want, rather than a more neutron-like or overly oscillatory profile in the SQK substrate language.

Once you’ve had a look, send me your impressions and I’ll help you do the next refinement pass.
......................
