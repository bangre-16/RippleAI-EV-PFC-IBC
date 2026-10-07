'use client'

import { Card } from '@/components/ui/card'
import { Spotlight } from '@/components/ui/spotlight'
import { SplineScene } from '@/components/ui/splite'

export function SplineSceneBasic() {
  return (
    <Card className="relative h-[420px] w-full overflow-hidden border-white/10 bg-black/[.75]">
      <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" size={280} />
      <div className="flex h-full flex-col md:flex-row">
        <div className="relative z-10 flex flex-1 flex-col justify-center p-7">
          <div className="text-[8px] font-semibold uppercase tracking-[.25em] text-cyan-100/55">Interactive 3D</div>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-[-.04em] text-white">Bring the charger to life.</h2>
          <p className="mt-3 max-w-md text-[10px] leading-6 text-white/40">A Spline-powered 3D scene adds a visual layer to the technical story while the live battery animation communicates the charging state.</p>
        </div>
        <div className="relative flex-1">
          <SplineScene scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode" className="h-full w-full" />
        </div>
      </div>
    </Card>
  )
}
