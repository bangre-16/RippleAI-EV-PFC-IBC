import { useEffect, useRef, useState, type ElementType, type ReactNode, type FormEvent } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  ArrowDownToLine,
  ArrowRight,
  BarChart3,
  BatteryCharging,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  Download,
  FileCode2,
  FileDown,
  FileText,
  Gauge,
  Layers3,
  LineChart,
  MonitorCog,
  Package,
  Play,
  Sparkles,
  UploadCloud,
  Zap,
} from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Spotlight } from '@/components/ui/spotlight'
import { SplineScene } from '@/components/ui/splite'
import { AnimatedBattery } from '@/components/AnimatedBattery'
import { cn } from '@/lib/utils'

const MODEL_ORDER = ['SVR', 'Gaussian Process', 'Random Forest', 'XGBoost', 'KNN', 'Stacked Ensemble'] as const
type ModelName = (typeof MODEL_ORDER)[number]
type Variant = 'baseline' | 'tuned'

// The Flask API stores Gaussian Process under the canonical key "GPR",
// while the UI displays the full name "Gaussian Process". Keep the mapping
// in one place so Models, Features and Results all read the correct data.
const DATA_KEY: Record<ModelName, string> = {
  'SVR': 'SVR',
  'Gaussian Process': 'GPR',
  'Random Forest': 'Random Forest',
  'XGBoost': 'XGBoost',
  'KNN': 'KNN',
  'Stacked Ensemble': 'Stacked Ensemble',
}

interface MetricSet { R2:number; MSE:number; RMSE:number; MAE:number; MAPE:number; MASE:number }
interface ImportanceItem { feature:string; label:string; importance:number; raw_drop?:number }
interface SiteData {
  features: string[]
  labels: Record<string,string>
  units: Record<string,string>
  rows: number
  columns: string[]
  ranges: Record<string,{min:number;max:number}>
  metrics: Record<string,{baseline:MetricSet;tuned:MetricSet}>
  hyper: Record<string,{baseline:unknown[];tuned:unknown[]}>
  importance: Record<string,{baseline:ImportanceItem[];tuned:ImportanceItem[]}>
  images: Record<string,string>
  model_zips: Record<string,string>
  simulation: {
    current:string; voltage:string; current_download:string; voltage_download:string;
    dataset_download:string; dataset_generation:string; slx:string; slxc:string
  }
  template_download:string
}

const fadeUp = {
  initial: { opacity:0, y:22 },
  animate: { opacity:1, y:0, transition:{ duration:.55, ease:[.2,.8,.2,1] } },
}

function useSiteData() {
  const [data,setData] = useState<SiteData | null>(null)
  const [error,setError] = useState('')
  useEffect(() => {
    fetch('/api/site-data')
      .then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(new Error(e.error || 'Could not load project data'))))
      .then(setData)
      .catch(e => setError(e.message))
  },[])
  return {data,error}
}

function Shell({ children }: { children: ReactNode }) {
  const location = useLocation()
  const nav = [
    ['/', 'Overview', Sparkles],
    ['/dataset', 'Dataset & Simulation', Database],
    ['/models', 'Models', Layers3],
    ['/feature-importance', 'Features', BarChart3],
    ['/results', 'Results', LineChart],
  ] as const
  return (
    <div className="min-h-screen overflow-x-hidden">
      <div className="fixed inset-0 -z-20 bg-[radial-gradient(circle_at_12%_4%,rgba(99,225,255,.08),transparent_28%),radial-gradient(circle_at_88%_12%,rgba(148,103,255,.09),transparent_30%),linear-gradient(145deg,#05070a,#0a0d11_48%,#07090c)]" />
      <div className="fixed inset-0 -z-10 subtle-grid opacity-[.20]" />
      <header className="sticky top-0 z-40 border-b border-white/8 bg-[#07090ccc]/80 backdrop-blur-2xl">
        <div className="container-page flex min-h-[72px] items-center justify-between gap-4">
          <Link to="/" className="flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-xl border border-cyan-200/20 bg-gradient-to-br from-cyan-200/20 to-violet-300/10 text-cyan-100 shadow-[0_0_40px_rgba(93,231,255,.09)]"><Zap className="h-5 w-5" /></span>
            <div><div className="font-display text-[15px] font-semibold tracking-[.13em] text-white">RIPPLE<span className="text-cyan-200">AI</span></div><div className="text-[9px] uppercase tracking-[.24em] text-white/35">EV PFC-IBC intelligence</div></div>
          </Link>
          <nav className="hidden items-center gap-1 xl:flex">
            {nav.map(([href,label,Icon]) => (
              <NavLink key={href} to={href} className={({isActive}) => cn('nav-link group flex items-center gap-2 rounded-full px-3 py-2 text-[10px] font-medium', isActive && 'active bg-white/[.045]')}>
                <Icon className="h-3.5 w-3.5 opacity-70 group-[.active]:opacity-100" />{label}
              </NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <NavLink to="/project-files" className={({isActive}) => cn('hidden rounded-full border border-white/10 px-3 py-2 text-[9px] font-semibold uppercase tracking-[.14em] text-white/55 transition hover:border-white/20 hover:text-white md:inline-flex', isActive && 'border-cyan-200/25 bg-white/[.04] text-white')}>Files</NavLink>
            <NavLink to="/prediction" className="group inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-cyan-200 to-violet-300 px-4 py-2.5 text-[10px] font-bold tracking-[.04em] text-slate-950 shadow-[0_12px_35px_rgba(106,221,251,.13)] transition hover:-translate-y-0.5 hover:shadow-[0_16px_45px_rgba(106,221,251,.18)]">Prediction Studio<ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" /></NavLink>
          </div>
        </div>
        <div className="container-page flex gap-1 overflow-x-auto pb-2 xl:hidden">
          {nav.map(([href,label,Icon]) => <NavLink key={href} to={href} className={({isActive}) => cn('nav-link flex shrink-0 items-center gap-2 rounded-full px-3 py-2 text-[9px]', isActive && 'active bg-white/[.05]')}><Icon className="h-3.5 w-3.5" />{label}</NavLink>)}
        </div>
      </header>
      <main className="container-page py-8 md:py-12">{children}</main>
      <footer className="container-page flex flex-col gap-3 border-t border-white/8 py-7 text-[9px] uppercase tracking-[.18em] text-white/28 sm:flex-row sm:items-center sm:justify-between"><span>RippleAI · PFC-IBC Current Ripple Intelligence</span><span>Simulation · ML · EV Battery Charging</span></footer>
      <div className="fixed bottom-5 left-1/2 z-40 hidden -translate-x-1/2 rounded-full border border-white/10 bg-black/45 px-4 py-2 text-[8px] uppercase tracking-[.22em] text-white/35 backdrop-blur-xl md:block">{location.pathname === '/' ? '01 / Overview' : location.pathname.slice(1).replaceAll('-', ' ')}</div>
    </div>
  )
}

function SectionTitle({ eyebrow, title, copy }: { eyebrow:string; title:string; copy?:string }) {
  return <motion.div {...fadeUp} className="mb-8 max-w-3xl"><div className="mb-3 flex items-center gap-2 text-[9px] font-semibold uppercase tracking-[.27em] text-cyan-200/70"><span className="h-px w-7 bg-gradient-to-r from-cyan-300 to-transparent" />{eyebrow}</div><h1 className="font-display text-4xl font-semibold tracking-[-.04em] text-white sm:text-5xl">{title}</h1>{copy && <p className="mt-4 max-w-2xl text-sm leading-7 text-white/45">{copy}</p>}</motion.div>
}

function StatCard({ icon:Icon, label, value, sub }: { icon:ElementType; label:string; value:string; sub:string }) {
  return <Card className="glass overflow-hidden border-white/8 bg-white/[.03] p-5"><div className="flex items-start justify-between"><Icon className="h-4 w-4 text-cyan-100/80" /><span className="text-[8px] uppercase tracking-[.2em] text-white/28">{label}</span></div><div className="mt-4 font-display text-3xl font-semibold text-white">{value}</div><p className="mt-1 text-[9px] leading-5 text-white/35">{sub}</p></Card>
}

function Overview({data}: {data:SiteData}) {
  return <div className="space-y-20">
    <section className="grid items-center gap-8 pt-2 lg:grid-cols-[1.02fr_.98fr]">
      <motion.div {...fadeUp} className="relative z-10">
        <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-cyan-200/15 bg-cyan-200/[.05] px-3 py-2 text-[9px] font-semibold uppercase tracking-[.22em] text-cyan-100/75"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-cyan-200" /> EV battery charging · PFC-IBC · ML prediction</div>
        <h1 className="font-display text-5xl font-semibold leading-[.98] tracking-[-.055em] text-white sm:text-6xl xl:text-7xl">Predict the ripple.<br /><span className="gradient-text">Understand the charger.</span></h1>
        <p className="mt-6 max-w-2xl text-[15px] leading-7 text-white/45">RippleAI is an end-to-end engineering workflow for estimating output current ripple in a PFC interleaved boost converter used in EV battery charging. Converter simulation becomes a structured dataset, machine learning models learn the nonlinear relationship, and the final web application turns those models into an interactive prediction tool.</p>
        <div className="mt-7 flex flex-wrap gap-3"><Link to="/prediction" className="group inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-200 to-violet-300 px-5 py-3 text-[10px] font-bold text-slate-950 transition hover:-translate-y-0.5"><Play className="h-3.5 w-3.5" />Open Prediction Studio<ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" /></Link><Link to="/dataset" className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[.035] px-5 py-3 text-[10px] font-semibold text-white/70 transition hover:bg-white/[.06]">Explore Dataset <ChevronRight className="h-3.5 w-3.5" /></Link></div>
        <div className="mt-10 grid max-w-2xl grid-cols-2 gap-3 sm:grid-cols-4"><StatCard icon={Database} label="Dataset" value={data.rows.toLocaleString()} sub="generated operating samples" /><StatCard icon={Gauge} label="Inputs" value="06" sub="converter parameters" /><StatCard icon={BrainCircuit} label="Models" value="06" sub="regression approaches" /><StatCard icon={Layers3} label="Stages" value="02" sub="baseline + tuned" /></div>
      </motion.div>
      <motion.div {...fadeUp} transition={{...fadeUp.animate.transition,delay:.08}} className="relative">
        <div className="absolute -inset-10 rounded-full bg-cyan-300/[.05] blur-3xl" />
        <AnimatedBattery />
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Card className="glass relative overflow-hidden p-3"><Spotlight size={180}/><div className="flex items-center gap-2 text-[8px] uppercase tracking-[.17em] text-white/32"><Activity className="h-3.5 w-3.5 text-cyan-200/70" />Ripple path</div><div className="mt-2 font-display text-sm text-white/80">Simulation → ML</div></Card>
          <Card className="glass relative overflow-hidden p-3"><div className="flex items-center gap-2 text-[8px] uppercase tracking-[.17em] text-white/32"><MonitorCog className="h-3.5 w-3.5 text-violet-200/70" />3D scene</div><div className="mt-2 font-display text-sm text-white/80">Interactive charger</div></Card>
        </div>
      </motion.div>
    </section>

    <section>
      <SectionTitle eyebrow="Project overview" title="A complete path from converter physics to usable prediction." copy="The work is built around one PFC interleaved boost converter, with the simulation supplying the data and the machine-learning layer learning the nonlinear current-ripple response." />
      <div className="grid gap-4 lg:grid-cols-4">{[
        ['01','Converter simulation','The PFC-IBC charging stage is simulated to obtain representative output-current and output-voltage behaviour.'],
        ['02','Automatic dataset','Six converter inputs are paired with simulated current-ripple targets across 2,000 operating samples.'],
        ['03','ML comparison','SVR, GPR, Random Forest, XGBoost, KNN and a stacked ensemble are evaluated before and after tuning.'],
        ['04','Interactive intelligence','The final website exposes prediction, feature influence, range checks, batch analysis and report generation.'],
      ].map(([n,t,c],i)=><motion.div key={n} {...fadeUp} transition={{...fadeUp.animate.transition,delay:i*.06}} className="glass rounded-2xl p-5"><div className="text-[9px] font-semibold tracking-[.2em] text-white/25">{n}</div><h3 className="mt-6 font-display text-xl text-white">{t}</h3><p className="mt-3 text-[11px] leading-6 text-white/40">{c}</p></motion.div>)}</div>
    </section>

    <section className="grid gap-5 lg:grid-cols-[1fr_.9fr]">
      <Card className="glass relative min-h-[390px] overflow-hidden p-0"><Spotlight size={320}/><div className="absolute inset-0 bg-gradient-to-br from-cyan-300/[.05] via-transparent to-violet-400/[.06]" /><div className="relative flex h-full flex-col justify-between p-7"><div><div className="text-[8px] font-semibold uppercase tracking-[.23em] text-cyan-100/55">Interactive 3D component</div><h2 className="mt-3 max-w-md font-display text-3xl font-semibold tracking-[-.035em]">A visual layer of 3D AI model.</h2><p className="mt-3 max-w-lg text-[11px] leading-6 text-white/40">The UI also includes an interactive Spline scene component, used as a visual entry point for the hardware-facing story while the animated 3D AI model.</p></div><div className="h-[210px] w-full overflow-hidden rounded-2xl border border-white/8 bg-black/20"><SplineScene scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode" className="h-full w-full" /></div></div></Card>
      <Card className="glass p-7"><div className="text-[8px] font-semibold uppercase tracking-[.23em] text-violet-200/60">Workflow</div><h2 className="mt-3 font-display text-3xl tracking-[-.035em]">From physics to prediction.</h2><div className="mt-7 space-y-4">{[
        ['Simulation','PFC-IBC waveforms'],['Dataset',`${data.rows.toLocaleString()} operating points`],['Training','Six regression models'],['Tuning','Optimized hyperparameters'],['Prediction','Ripple estimate'],
      ].map(([t,s],i)=><div key={t} className="flex items-center gap-4"><div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl border border-white/8 bg-white/[.025] font-display text-[11px] text-cyan-100/75">0{i+1}</div><div className="flex-1"><div className="font-semibold text-[11px] text-white/80">{t}</div><div className="text-[9px] text-white/32">{s}</div></div>{i<4 && <ArrowDownToLine className="h-3.5 w-3.5 text-white/15" />}</div>)}
      </div></Card>
    </section>
  </div>
}

function DatasetPage({data}: {data:SiteData}) {
  const preview = data.columns.slice(0,7)
  return <div className="space-y-10"><SectionTitle eyebrow="02 · Dataset & simulation" title="The simulated operating space behind every prediction." copy="This page brings together the converter simulation outputs, the generated dataset, operating ranges, and downloadable project files." />
    <div className="grid gap-4 md:grid-cols-4"><StatCard icon={Database} label="Rows" value={data.rows.toLocaleString()} sub="simulation-derived samples"/><StatCard icon={Layers3} label="Columns" value={String(data.columns.length).padStart(2,'0')} sub="six inputs + one target"/><StatCard icon={Zap} label="Target" value="ΔIout" sub="current ripple in amperes"/><StatCard icon={Gauge} label="Coverage" value="2,000" sub="generated operating points"/></div>
    <div className="grid gap-5 lg:grid-cols-2"><SimulationCard title="Output current waveform" image={data.simulation.current} download={data.simulation.current_download} icon={Activity} /><SimulationCard title="Output voltage waveform" image={data.simulation.voltage} download={data.simulation.voltage_download} icon={Gauge} /></div>
    <div className="grid gap-5 lg:grid-cols-[1.15fr_.85fr]"><Card className="glass p-6"><div className="flex items-start justify-between gap-5"><div><div className="text-[8px] uppercase tracking-[.23em] text-cyan-100/50">Dataset preview</div><h2 className="mt-2 font-display text-2xl">Six inputs + ripple target.</h2></div><a href={data.simulation.dataset_download} download className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-[9px] font-semibold text-white/60 transition hover:bg-white/[.05]"><Download className="h-3.5 w-3.5"/> dataset.csv</a></div><div className="mt-6 overflow-auto rounded-xl border border-white/8"><table className="w-full min-w-[820px] text-left text-[9px]"><thead className="bg-white/[.03] text-white/40"><tr>{preview.map(c=><th key={c} className="px-3 py-3 font-semibold">{c.replaceAll('_',' ')}</th>)}</tr></thead><tbody>{Array.from({length:6}).map((_,i)=><tr key={i} className="border-t border-white/6">{preview.map(c=><td key={c} className="px-3 py-3 text-white/45">{(data.ranges[c] ? (data.ranges[c].min + (data.ranges[c].max-data.ranges[c].min)*(i+1)/7).toFixed(3) : '—')}</td>)}</tr>)}</tbody></table></div></Card><Card className="glass p-6"><div className="text-[8px] uppercase tracking-[.23em] text-violet-200/55">Operating ranges</div><h2 className="mt-2 font-display text-2xl">Training envelope.</h2><div className="mt-5 space-y-3">{data.features.map(f=><div key={f} className="rounded-xl border border-white/7 bg-white/[.025] p-3"><div className="flex items-center justify-between gap-4"><span className="text-[10px] font-semibold text-white/70">{data.labels[f]}</span><span className="text-[8px] text-white/30">{data.units[f]}</span></div><div className="mt-2 font-mono text-[9px] text-cyan-100/55">{data.ranges[f].min.toFixed(2)} — {data.ranges[f].max.toFixed(2)}</div></div>)}</div></Card></div>
    <div><div className="mb-4 text-[8px] uppercase tracking-[.23em] text-white/25">Downloads</div><div className="grid gap-3 md:grid-cols-4"><DownloadTile label="dataset.csv" href={data.simulation.dataset_download} icon={Database}/><DownloadTile label="dataset_generation.m" href={data.simulation.dataset_generation} icon={FileCode2}/><DownloadTile label="PFC_IBC_AC_Model.slx" href={data.simulation.slx} icon={Package}/><DownloadTile label="PFC_IBC_AC_Model.slxc" href={data.simulation.slxc} icon={Package}/></div></div>
  </div>
}

function SimulationCard({title,image,download,icon:Icon}:{title:string;image:string;download:string;icon:ElementType}) {
  return <Card className="glass overflow-hidden"><div className="flex items-center justify-between gap-4 border-b border-white/8 p-5"><div className="flex items-center gap-3"><span className="grid h-9 w-9 place-items-center rounded-xl border border-white/8 bg-white/[.035]"><Icon className="h-4 w-4 text-cyan-100/75"/></span><div><div className="font-display text-lg text-white/90">{title}</div><div className="text-[8px] uppercase tracking-[.17em] text-white/25">actual simulation output</div></div></div><a href={download} download className="inline-flex items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-[8px] text-white/55 hover:bg-white/[.05]"><Download className="h-3.5 w-3.5"/> Download</a></div><div className="bg-black/20 p-4"><img src={image} alt={title} className="h-[330px] w-full rounded-xl border border-white/7 object-contain bg-black/20" /></div></Card>
}
function DownloadTile({label,href,icon:Icon}:{label:string;href:string;icon:ElementType}){return <a href={href} download className="glass group flex items-center justify-between gap-3 rounded-xl p-4 transition hover:-translate-y-0.5 hover:bg-white/[.055]"><div className="flex items-center gap-3 min-w-0"><span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-white/[.04]"><Icon className="h-4 w-4 text-cyan-100/65"/></span><span className="truncate text-[10px] font-medium text-white/65">{label}</span></div><Download className="h-3.5 w-3.5 shrink-0 text-white/25 transition group-hover:text-cyan-100"/></a>}

function ModelsPage({data}:{data:SiteData}) {
  const [variant,setVariant]=useState<Variant>('baseline')
  return <div className="space-y-8"><SectionTitle eyebrow="03 · Models" title="Six regressors. Two model stages. One target." copy="Every model page uses the saved project artifacts. Baseline and tuned variants are presented side by side through the same visual system." /><div className="glass sticky top-[84px] z-20 flex w-fit rounded-full p-1"><button onClick={()=>setVariant('baseline')} className={cn('rounded-full px-4 py-2 text-[9px] uppercase tracking-[.16em] transition',variant==='baseline'?'bg-white text-slate-950':'text-white/45')}>Without tuning</button><button onClick={()=>setVariant('tuned')} className={cn('rounded-full px-4 py-2 text-[9px] uppercase tracking-[.16em] transition',variant==='tuned'?'bg-gradient-to-r from-cyan-200 to-violet-300 text-slate-950':'text-white/45')}>With tuning</button></div><div className="grid gap-5 xl:grid-cols-2">{MODEL_ORDER.map((name,i)=><ModelCard key={name} name={name} data={data} variant={variant} index={i}/>)}</div></div>
}
function ModelCard({name,data,variant,index}:{name:ModelName;data:SiteData;variant:Variant;index:number}){
  const key=DATA_KEY[name]; const metrics=data.metrics[key]; const m=metrics[variant]; const img=data.images[`${name}|${variant}`]; const zip=data.model_zips[`${key}|${variant}`];
  return <motion.div {...fadeUp} transition={{...fadeUp.animate.transition,delay:index*.05}}><Card className="glass overflow-hidden"><div className="flex flex-col gap-5 p-6 md:flex-row md:items-start md:justify-between"><div className="max-w-xl"><div className="flex items-center gap-2"><span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-200/15 to-violet-300/10"><Cpu className="h-4 w-4 text-cyan-100/70"/></span><div><h2 className="font-display text-2xl text-white">{name}</h2><p className="text-[8px] uppercase tracking-[.17em] text-white/25">{variant==='baseline'?'baseline model':'optimized model'}</p></div></div><p className="mt-4 text-[10px] leading-6 text-white/40">Saved regression model for current-ripple estimation using the converter operating parameters.</p></div><a href={`/static/downloads/${zip}`} download className="inline-flex shrink-0 items-center gap-2 rounded-xl border border-white/10 bg-white/[.025] px-4 py-2.5 text-[9px] font-semibold text-white/60 transition hover:border-cyan-200/25 hover:text-white"><FolderArchiveIcon/> Download folder</a></div><div className="grid gap-4 border-t border-white/8 p-5 md:grid-cols-[1fr_1.1fr]"><div className="grid grid-cols-3 gap-2 content-start">{[['R²',m.R2.toFixed(4)],['RMSE',m.RMSE.toFixed(5)],['MAE',m.MAE.toFixed(5)]].map(([k,v])=><div key={k} className="rounded-xl border border-white/7 bg-white/[.025] p-3"><div className="text-[8px] uppercase tracking-[.18em] text-white/25">{k}</div><div className="mt-2 font-display text-lg text-white/80">{v}</div></div>)}<div className="col-span-3 rounded-xl border border-cyan-200/10 bg-cyan-200/[.025] p-4"><div className="text-[8px] uppercase tracking-[.18em] text-cyan-100/45">All metrics</div><div className="mt-3 grid grid-cols-3 gap-3 text-[9px] text-white/45"><span>MSE <b className="ml-1 text-white/75">{m.MSE.toFixed(6)}</b></span><span>MAPE <b className="ml-1 text-white/75">{m.MAPE.toFixed(2)}%</b></span><span>MASE <b className="ml-1 text-white/75">{m.MASE.toFixed(3)}</b></span></div></div></div><div className="overflow-hidden rounded-xl border border-white/8 bg-black/20"><img src={`/static/images/${img}`} alt={`${name} ${variant} actual versus predicted`} className="h-[300px] w-full object-contain" /></div></div><div className="border-t border-white/8 px-6 py-4 text-[8px] uppercase tracking-[.18em] text-white/25">Saved artifact · actual vs predicted graph · {variant==='baseline'?'original':'tuned'} configuration</div></Card></motion.div>
}
function FolderArchiveIcon(){return <Package className="h-3.5 w-3.5"/>}

function FeaturesPage({data}:{data:SiteData}) {
  const [variant,setVariant]=useState<Variant>('baseline')
  return <div className="space-y-8"><SectionTitle eyebrow="04 · Feature influence" title="Which inputs shape the model prediction?" copy="Relative influence is shown for each model and should be read as model-based influence, not as a direct physical percentage contribution to converter ripple."/><div className="glass flex w-fit rounded-full p-1"><button onClick={()=>setVariant('baseline')} className={cn('rounded-full px-4 py-2 text-[9px] uppercase tracking-[.16em]',variant==='baseline'?'bg-white text-slate-950':'text-white/45')}>Without tuning</button><button onClick={()=>setVariant('tuned')} className={cn('rounded-full px-4 py-2 text-[9px] uppercase tracking-[.16em]',variant==='tuned'?'bg-gradient-to-r from-cyan-200 to-violet-300 text-slate-950':'text-white/45')}>With tuning</button></div><div className="grid gap-5 lg:grid-cols-2">{MODEL_ORDER.map((name,i)=><FeatureCard key={name} name={name} items={data.importance[DATA_KEY[name]][variant]} index={i} variant={variant}/>)}</div></div>
}
function FeatureCard({name,items,index,variant}:{name:string;items:ImportanceItem[];index:number;variant:Variant}){
  const sorted=[...items].sort((a,b)=>b.importance-a.importance)
  return <motion.div {...fadeUp} transition={{...fadeUp.animate.transition,delay:index*.04}}><Card className="glass p-6"><div className="flex items-center justify-between"><div><div className="text-[8px] uppercase tracking-[.2em] text-cyan-100/45">{variant==='baseline'?'baseline':'tuned'} influence</div><h2 className="mt-1 font-display text-2xl text-white">{name}</h2></div><BarChart3 className="h-4 w-4 text-white/25"/></div><div className="mt-6 space-y-3">{sorted.map((item,j)=><div key={item.feature}><div className="mb-1.5 flex items-center justify-between gap-4"><span className="text-[9px] font-medium text-white/65">{item.label}</span><span className="font-mono text-[9px] text-white/42">{item.importance.toFixed(2)}%</span></div><div className="h-2 rounded-full bg-white/[.055]"><motion.div initial={{width:0}} whileInView={{width:`${Math.min(item.importance,100)}%`}} viewport={{once:true}} transition={{duration:.75,delay:j*.05}} className="h-full rounded-full bg-gradient-to-r from-cyan-200/80 to-violet-300/80" /></div></div>)}</div><div className="mt-5 border-t border-white/8 pt-4 text-[8px] leading-5 text-white/25">Highest relative influence: <span className="text-cyan-100/60">{sorted[0]?.label}</span></div></Card></motion.div>
}

function ResultsPage({data}:{data:SiteData}) {
  const metricKeys=['R2','MSE','RMSE','MAE','MAPE','MASE'] as const
  return <div className="space-y-10"><SectionTitle eyebrow="05 · Results & comparison" title="Baseline versus tuned performance." copy="Use the tables for numerical comparison and the saved actual-versus-predicted figures for visual validation."/><ResultsTable title="Without tuning" data={data} variant="baseline" metricKeys={metricKeys}/><ResultsTable title="With tuning" data={data} variant="tuned" metricKeys={metricKeys}/><div><div className="mb-4 text-[8px] uppercase tracking-[.22em] text-white/25">Visual comparison</div><div className="grid gap-5 xl:grid-cols-2">{MODEL_ORDER.map(name=><Card key={name} className="glass overflow-hidden"><div className="flex items-center justify-between border-b border-white/8 px-5 py-4"><div className="font-display text-lg">{name}</div><span className="text-[8px] uppercase tracking-[.16em] text-white/25">baseline + tuned</span></div><div className="grid gap-px bg-white/6 md:grid-cols-2"><img src={`/static/images/${data.images[`${name}|baseline`]}`} alt={`${name} baseline`} className="h-[250px] w-full bg-black/20 object-contain"/><img src={`/static/images/${data.images[`${name}|tuned`]}`} alt={`${name} tuned`} className="h-[250px] w-full bg-black/20 object-contain"/></div></Card>)}</div></div></div>
}
function ResultsTable({title,data,variant,metricKeys}:{title:string;data:SiteData;variant:Variant;metricKeys:readonly string[]}){
  return <Card className="glass overflow-hidden"><div className="flex items-center justify-between gap-4 border-b border-white/8 px-6 py-5"><div><div className="text-[8px] uppercase tracking-[.22em] text-white/25">Model evaluation</div><h2 className="mt-1 font-display text-2xl">{title}</h2></div><div className={cn('rounded-full px-3 py-2 text-[8px] uppercase tracking-[.18em]',variant==='tuned'?'bg-cyan-200/10 text-cyan-100/65':'bg-white/[.04] text-white/35')}>{variant==='tuned'?'optimized':'original'}</div></div><div className="overflow-auto"><table className="w-full min-w-[980px] text-left text-[9px]"><thead className="bg-white/[.025] text-white/35"><tr><th className="px-4 py-3">Model</th>{metricKeys.map(k=><th key={k} className="px-4 py-3">{k==='R2'?'R²':k}</th>)}</tr></thead><tbody>{MODEL_ORDER.map((name)=><tr key={name} className="border-t border-white/6"><td className="px-4 py-4 font-semibold text-white/75">{name}</td>{metricKeys.map(k=><td key={k} className="px-4 py-4 font-mono text-white/43">{k==='R2'?data.metrics[DATA_KEY[name]][variant].R2.toFixed(6):k==='MAPE'?`${data.metrics[DATA_KEY[name]][variant][k].toFixed(3)}%`:data.metrics[DATA_KEY[name]][variant][k].toFixed(6)}</td>)}</tr>)}</tbody></table></div></Card>
}

function PredictionPage({data}:{data:SiteData}) {
  const defaults=Object.fromEntries(data.features.map(f=>[f,(data.ranges[f].min+data.ranges[f].max)/2])) as Record<string,number>
  const [values,setValues]=useState<Record<string,number>>(defaults)
  const [pred,setPred]=useState<Record<string,number|null>|null>(null)
  const [warnings,setWarnings]=useState<string[]>([])
  const [recommended,setRecommended]=useState<string>('—')
  const [status,setStatus]=useState('Ready for prediction.')
  const [batchFile,setBatchFile]=useState<File|null>(null)
  const [batch,setBatch]=useState<{columns:string[];rows:Record<string,unknown>[];csv_base64:string;row_count:number;out_of_range_cells:number;recommended:string|null}|null>(null)
  const [batchStatus,setBatchStatus]=useState('Waiting for a CSV file.')
  const [reportStatus,setReportStatus]=useState('Run a prediction or batch analysis to enable report generation.')
  const [reportReady,setReportReady]=useState(false)
  const fileInput=useRef<HTMLInputElement>(null)

  const runSingle=async(e:FormEvent)=>{
    e.preventDefault();setStatus('Running all saved tuned models…');setPred(null);setReportReady(false)
    const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});const json=await r.json()
    if(!r.ok||!json.success){setStatus(json.error||'Prediction failed.');return}
    setPred(json.predictions);setWarnings(json.warnings||[]);setRecommended(json.recommended||'—');setStatus('Prediction complete.');setReportReady(true);setReportStatus('Prediction report is ready to generate.')
  }

  const runBatch=async()=>{
    if(!batchFile)return;setBatchStatus('Validating CSV and running all tuned models…');setBatch(null);setReportReady(false)
    const fd=new FormData();fd.append('file',batchFile)
    const r=await fetch('/api/batch-predict',{method:'POST',body:fd});const json=await r.json();if(!r.ok||!json.success){setBatchStatus(json.error||'Batch prediction failed.');return}
    setBatch(json);setBatchStatus(`Completed ${json.row_count} operating points.`);setReportReady(true);setReportStatus('Batch report is ready. Generate the PDF to capture the complete batch output.');setPred(null)
  }

  const downloadBatch=()=>{if(!batch?.csv_base64)return;const binary=atob(batch.csv_base64);const bytes=Uint8Array.from(binary,c=>c.charCodeAt(0));const blob=new Blob([bytes],{type:'text/csv;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='rippleai_batch_predictions.csv';a.click();URL.revokeObjectURL(url)}

  const generateReport=async()=>{
    if(!reportReady)return;setReportStatus('Generating PDF report…')
    let payload:any
    if(batch){payload={report_type:'batch',csv_base64:batch.csv_base64,recommended:batch.recommended,row_count:batch.row_count,out_of_range_cells:batch.out_of_range_cells}}
    else payload={inputs:values,predictions:pred,warnings,recommended}
    const r=await fetch('/api/generate-report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    if(!r.ok){const j=await r.json().catch(()=>({}));setReportStatus(j.error||'Report generation failed.');return}
    const blob=await r.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=batch?'RippleAI_Batch_Prediction_Report.pdf':'RippleAI_Prediction_Report.pdf';a.click();URL.revokeObjectURL(url);setReportStatus('PDF generated and downloaded.')
  }

  return <div className="space-y-8"><SectionTitle eyebrow="06 · Prediction Studio" title="Turn an operating point into a ripple estimate." copy="The backend loads the saved tuned model artifacts. No model is retrained for an individual query."/><div className="grid gap-5 xl:grid-cols-[1.05fr_.95fr]"><Card className="glass p-6"><div className="text-[8px] uppercase tracking-[.22em] text-cyan-100/50">Single prediction</div><h2 className="mt-2 font-display text-2xl">Converter specifications</h2><form onSubmit={runSingle} className="mt-6 space-y-4">{data.features.map(f=><label key={f} className="block"><div className="flex items-center justify-between gap-3"><span className="text-[10px] font-semibold text-white/65">{data.labels[f]}</span><span className="text-[8px] text-white/25">{data.ranges[f].min.toFixed(2)} — {data.ranges[f].max.toFixed(2)} {data.units[f]}</span></div><div className="mt-2 flex rounded-xl border border-white/8 bg-white/[.025]"><input type="number" step="any" value={values[f]} onChange={e=>setValues(v=>({...v,[f]:Number(e.target.value)}))} className="w-full border-0 bg-transparent px-4 py-3 text-[11px] text-white focus:ring-0"/><span className="grid min-w-[66px] place-items-center border-l border-white/8 px-2 text-[8px] text-white/30">{data.units[f]}</span></div></label>)}<button className="group mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-200 to-violet-300 px-5 py-3 text-[10px] font-bold text-slate-950">Run all tuned models <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-1"/></button></form></Card><div className="space-y-4"><Card className="glass p-6"><div className="flex items-start justify-between"><div><div className="text-[8px] uppercase tracking-[.22em] text-white/25">Model outputs</div><div className="mt-2 text-[9px] text-white/35">{status}</div></div><Activity className="h-4 w-4 text-cyan-100/55"/></div>{warnings.length>0&&<div className="mt-4 rounded-xl border border-amber-200/15 bg-amber-200/[.05] p-4 text-[9px] leading-5 text-amber-100/70">{warnings.map(w=><div key={w} className="flex gap-2"><AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0"/>{w}</div>)}</div>}<div className="mt-5 space-y-2">{pred?Object.entries(pred).map(([name,v])=><div key={name} className="flex items-center justify-between rounded-xl border border-white/7 bg-white/[.025] p-4"><div><div className="text-[10px] font-semibold text-white/70">{name}</div>{name===recommended&&<div className="mt-1 text-[8px] text-cyan-100/55">selected from saved tuned-model evaluation</div>}</div><div className="font-display text-lg text-white">{v==null?'—':Number(v).toFixed(6)} <span className="text-[9px] text-white/35">A</span></div></div>):<div className="rounded-xl border border-dashed border-white/10 p-10 text-center text-[9px] text-white/25">Run the model to see all predictions.</div>}</div></Card><Card className="glass bg-gradient-to-br from-cyan-200/[.05] to-violet-300/[.05] p-6"><div className="text-[8px] uppercase tracking-[.22em] text-cyan-100/45">Recommendation</div><div className="mt-2 font-display text-3xl text-white">{recommended}</div><p className="mt-2 text-[9px] leading-5 text-white/32">Based on stored tuned-model evaluation performance. It is not selected from the unknown true ripple value.</p></Card></div></div>

    <div className="grid gap-5 xl:grid-cols-[1.1fr_.9fr]"><Card className="glass p-6"><div className="flex items-start justify-between"><div><div className="text-[8px] uppercase tracking-[.22em] text-cyan-100/50">Batch prediction</div><h2 className="mt-2 font-display text-2xl">Upload many operating points at once.</h2><p className="mt-2 text-[10px] leading-6 text-white/35">Use a CSV containing the six required converter inputs. Every row is processed through the saved tuned models.</p></div><UploadCloud className="h-5 w-5 text-cyan-100/65"/></div><div onClick={()=>fileInput.current?.click()} className="mt-5 cursor-pointer rounded-2xl border border-dashed border-white/12 bg-white/[.02] p-8 text-center transition hover:border-cyan-200/25 hover:bg-cyan-200/[.025]"><input ref={fileInput} type="file" accept=".csv,text/csv" onChange={e=>setBatchFile(e.target.files?.[0]||null)} className="hidden"/><UploadCloud className="mx-auto h-7 w-7 text-cyan-100/45"/><div className="mt-3 font-semibold text-[11px] text-white/65">{batchFile?batchFile.name:'Choose a CSV file'}</div><div className="mt-1 text-[8px] text-white/25">Six required inputs · up to 5,000 rows</div></div><div className="mt-4 flex flex-wrap items-center gap-3"><a href={data.template_download} download className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-4 py-2.5 text-[9px] text-white/55 hover:bg-white/[.05]"><Download className="h-3.5 w-3.5"/> CSV template</a><button onClick={runBatch} disabled={!batchFile} className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-[9px] font-bold text-slate-950 disabled:cursor-not-allowed disabled:opacity-35">Run batch prediction<ArrowRight className="h-3.5 w-3.5"/></button><span className="text-[8px] text-white/25">{batchStatus}</span></div>{batch&&<div className="mt-5"><div className="grid grid-cols-3 gap-2"><MiniStat label="Rows" value={batch.row_count}/><MiniStat label="Out-of-range cells" value={batch.out_of_range_cells}/><MiniStat label="Recommendation" value={batch.recommended||'—'}/></div><div className="mt-4 overflow-auto rounded-xl border border-white/8"><table className="min-w-[980px] w-full text-left text-[8px]"><thead className="bg-white/[.03] text-white/35"><tr>{batch.columns.map(c=><th key={c} className="px-3 py-3">{c.replaceAll('_',' ')}</th>)}</tr></thead><tbody>{batch.rows.slice(0,80).map((row,i)=><tr key={i} className="border-t border-white/6">{batch.columns.map(c=><td key={c} className="px-3 py-2.5 text-white/45">{typeof row[c]==='number'?Number(row[c]).toFixed(5):String(row[c])}</td>)}</tr>)}</tbody></table></div><button onClick={downloadBatch} className="mt-3 inline-flex items-center gap-2 rounded-xl border border-white/10 px-4 py-2.5 text-[9px] font-semibold text-white/55 hover:bg-white/[.05]"><Download className="h-3.5 w-3.5"/> Download predictions CSV</button></div>}</Card>
      <Card className="glass p-6 bg-gradient-to-br from-violet-300/[.055] to-cyan-200/[.035]"><div className="text-[8px] uppercase tracking-[.22em] text-violet-200/60">Automatic report generation</div><h2 className="mt-2 font-display text-2xl">Turn the latest analysis into a PDF.</h2><p className="mt-3 text-[10px] leading-6 text-white/35">Generate a shareable report containing the inputs, model outputs, range status, recommendation, feature-influence context and prediction comparison graph. For batch mode, the complete uploaded batch output is included.</p><div className="mt-5 grid grid-cols-2 gap-2">{['Operating inputs','Model predictions','Range status','Feature influence','Prediction graph','Batch results'].map(t=><div key={t} className="flex items-center gap-2 rounded-xl border border-white/7 bg-white/[.025] p-3 text-[9px] text-white/45"><Check className="h-3.5 w-3.5 text-cyan-100/70"/>{t}</div>)}</div><button onClick={generateReport} disabled={!reportReady} className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-violet-200 to-cyan-200 px-5 py-3 text-[10px] font-bold text-slate-950 disabled:cursor-not-allowed disabled:opacity-30"><FileDown className="h-4 w-4"/>Generate PDF report</button><div className="mt-3 text-center text-[8px] text-white/25">{reportStatus}</div></Card></div>
  </div>
}
function MiniStat({label,value}:{label:string;value:unknown}){return <div className="rounded-xl border border-white/7 bg-white/[.025] p-3"><span className="text-[7px] uppercase tracking-[.16em] text-white/25">{label}</span><strong className="mt-2 block truncate font-display text-sm text-white/75">{String(value)}</strong></div>}

function FilesPage({data}:{data:SiteData}) {
  return <div className="space-y-8"><SectionTitle eyebrow="07 · Project files" title="Everything needed to inspect and reproduce the workflow." copy="Download source code, simulation assets, datasets and model artifacts from the same interface."/><div className="grid gap-5 lg:grid-cols-2"><Card className="glass p-6"><div className="text-[8px] uppercase tracking-[.22em] text-cyan-100/45">Source code</div><div className="mt-5 space-y-2">{['Dataset__dataset_generation.m','SVR__train_svr.py','SVR Tuned__tune_svr.py','Gaussian Process__train_gpr.py','Gaussian Process Tuned__tune_gpr.py','Random Forest__train_random_forest.py','RF Tuned__tune_random_forest.py','XGBoost__train_xgboost.py','XGBoost Tuned__tune_xgboost.py','KNN__train_knn.py','KNN Tuned__tune_knn.py','Stacked Meta Ensemble Tuned__train_stacked_ensemble.py'].map(f=><a key={f} href={`/static/code/${encodeURIComponent(f)}`} download className="group flex items-center justify-between gap-3 rounded-xl border border-white/7 bg-white/[.025] p-3"><div className="flex min-w-0 items-center gap-3"><FileCode2 className="h-4 w-4 shrink-0 text-cyan-100/55"/><span className="truncate text-[9px] text-white/55">{f}</span></div><Download className="h-3.5 w-3.5 shrink-0 text-white/20 group-hover:text-white/70"/></a>)}</div></Card><Card className="glass p-6"><div className="text-[8px] uppercase tracking-[.22em] text-violet-200/55">Project artifacts</div><div className="mt-5 grid gap-2 sm:grid-cols-2"><DownloadTile label="dataset.csv" href={data.simulation.dataset_download} icon={Database}/><DownloadTile label="Simulation model .slx" href={data.simulation.slx} icon={Package}/><DownloadTile label="Current waveform" href={data.simulation.current_download} icon={Activity}/><DownloadTile label="Voltage waveform" href={data.simulation.voltage_download} icon={Gauge}/><DownloadTile label="Batch template" href={data.template_download} icon={UploadCloud}/></div><div className="mt-6 rounded-2xl border border-cyan-200/10 bg-cyan-200/[.025] p-5"><div className="flex items-center gap-2 text-[8px] uppercase tracking-[.2em] text-cyan-100/55"><CheckCircle2 className="h-3.5 w-3.5"/> Core workflow</div><p className="mt-2 text-[10px] leading-6 text-white/35">Simulation → dataset → baseline models → tuned models → feature influence → prediction → batch prediction → report.</p></div></Card></div></div>
}

function App() {
  const location = useLocation()
  const {data,error}=useSiteData()
  if(error) return <div className="grid min-h-screen place-items-center px-6 text-center"><div><div className="font-display text-2xl">Could not load RippleAI data.</div><p className="mt-2 text-sm text-white/40">{error}</p></div></div>
  if(!data) return <div className="grid min-h-screen place-items-center"><div className="rounded-full border border-white/10 bg-white/[.03] px-5 py-3 text-[9px] uppercase tracking-[.22em] text-white/45 animate-pulse">Loading RippleAI</div></div>
  return <Shell><AnimatePresence mode="wait"><motion.div key={location.pathname} initial={{opacity:0,y:14}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-10}} transition={{duration:.3}}><Routes><Route path="/" element={<Overview data={data}/>} /><Route path="/dataset" element={<DatasetPage data={data}/>} /><Route path="/models" element={<ModelsPage data={data}/>} /><Route path="/feature-importance" element={<FeaturesPage data={data}/>} /><Route path="/results" element={<ResultsPage data={data}/>} /><Route path="/prediction" element={<PredictionPage data={data}/>} /><Route path="/project-files" element={<FilesPage data={data}/>} /></Routes></motion.div></AnimatePresence></Shell>
}

export default App
