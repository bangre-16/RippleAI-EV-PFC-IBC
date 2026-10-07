import { motion } from 'framer-motion'
import { BatteryCharging, Zap } from 'lucide-react'

export function AnimatedBattery() {
  return (
    <div className="relative flex min-h-[420px] items-center justify-center overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.025] p-8 shadow-[0_35px_100px_rgba(0,0,0,.35)]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_45%,rgba(91,225,255,.13),transparent_32%),radial-gradient(circle_at_75%_25%,rgba(145,106,255,.16),transparent_30%)]" />
      <div className="absolute h-[280px] w-[280px] rounded-full border border-cyan-300/10" />
      <div className="absolute h-[360px] w-[360px] rounded-full border border-violet-300/10" />
      <div className="relative w-[280px]">
        <div className="absolute -top-7 left-1/2 h-5 w-16 -translate-x-1/2 rounded-t-xl border border-white/15 bg-white/10" />
        <div className="relative overflow-hidden rounded-[2rem] border border-white/20 bg-black/60 p-4 shadow-[0_0_70px_rgba(89,227,255,.12)] backdrop-blur-xl">
          <div className="absolute inset-0 bg-gradient-to-br from-white/[0.08] via-transparent to-violet-400/[0.08]" />
          <div className="relative flex items-center justify-between border-b border-white/10 px-2 pb-4">
            <span className="text-[10px] font-semibold uppercase tracking-[0.26em] text-white/45">EV Battery · UI Demo</span>
            <BatteryCharging className="h-4 w-4 text-cyan-200" />
          </div>
          <div className="relative mt-5 rounded-2xl border border-white/10 bg-black/50 p-3">
            <div className="relative h-28 overflow-hidden rounded-xl border border-white/10 bg-white/[0.03]">
              <motion.div
                className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-cyan-300/80 via-cyan-300/45 to-violet-400/30"
                animate={{ height: ['15%', '86%', '48%', '92%'] }}
                transition={{ duration: 5.8, repeat: Infinity, ease: 'easeInOut' }}
              />
              <motion.div
                className="absolute inset-x-0 bottom-0 h-full bg-[linear-gradient(180deg,transparent,rgba(255,255,255,.08),transparent)]"
                animate={{ y: ['-100%', '100%'] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: 'linear' }}
              />
              <div className="absolute inset-0 grid grid-cols-4 opacity-30">
                <span className="border-r border-white/10" /><span className="border-r border-white/10" /><span className="border-r border-white/10" />
              </div>
            </div>
            <div className="mt-4 flex items-end justify-between">
              <div>
                <div className="text-[9px] uppercase tracking-[0.22em] text-white/35">Animated charge demo</div>
                <div className="mt-1 font-display text-3xl font-semibold text-white">87<span className="text-cyan-200">%</span></div>
              </div>
              <motion.div animate={{ scale: [1, 1.06, 1], opacity: [0.7, 1, 0.7] }} transition={{ duration: 2.2, repeat: Infinity }} className="flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-2 text-[9px] font-semibold uppercase tracking-[0.18em] text-cyan-100">
                <Zap className="h-3.5 w-3.5" /> Charging
              </motion.div>
            </div>
          </div>
          <div className="relative mt-5 grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><span className="text-[8px] uppercase tracking-[0.2em] text-white/35">DC link</span><strong className="mt-1 block font-display text-sm">850 V</strong></div>
            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3"><span className="text-[8px] uppercase tracking-[0.2em] text-white/35">Ripple</span><strong className="mt-1 block font-display text-sm text-cyan-100">0.42 A</strong></div>
          </div>
        </div>
        <motion.div className="absolute -left-10 top-1/2 h-px w-10 bg-gradient-to-r from-transparent to-cyan-300" animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.8, repeat: Infinity }} />
        <motion.div className="absolute -right-10 top-[58%] h-px w-10 bg-gradient-to-r from-violet-300 to-transparent" animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 2.1, repeat: Infinity, delay: .4 }} />
      </div>
    </div>
  )
}
