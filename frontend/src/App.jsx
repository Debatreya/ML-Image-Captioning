

function App() {

  return (
    <div className="grid place-items-center h-[100vh]">
      <div className="h-[480px] w-[850px] shadow-xl hover:shadow-3xl border-color : ring">
        <img src="https://dfstudio-d420.kxcdn.com/wordpress/wp-content/uploads/2019/06/digital_camera_photo-1080x675.jpg" alt="" className="w-full h-full object-cover"/>
      </div>
      <button className="inline-flex h-8 min-h-8 flex-shrink-0 pl-4 pr-4 text-center font-medium mt-[-8rem] border border-slate-600 hover:text-white hover:bg-slate-700">Upload photo</button>
    </div>
  )
}

export default App
