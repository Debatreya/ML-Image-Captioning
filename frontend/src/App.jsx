import axios from "axios";
import { useState } from "react";
import fs from "fs";

function App() {

  const [image, setImage] = useState();

	const handleSubmit = async (e) => {
		e.preventDefault();
		const formData = new FormData();
		const image = document.querySelector("input[name=image]").files[0];
		formData.append("image", image);
		try {
			const res = await axios.post("http://localhost:5000/runModel", formData);
      // Store image in folder
      fs.writeFileSync('./Output/output.jpg', res.data); // This will store image in Output folder
			setImage('./Output/output.jpg');
		} catch (err) {
			console.log(err);
		}
	};

	return (
		<div className="grid place-items-center h-[100vh]">
			<div className="h-[480px] w-[850px] shadow-xl hover:shadow-3xl border-color : ring">
				{image && <img
					src={image}
					alt=""
					className="w-full h-full object-cover"
				/>}
			</div>
			<form>
				<input type="file" name="image" />
				<button
					className="inline-flex h-8 min-h-8 flex-shrink-0 pl-4 pr-4 text-center font-medium mt-[-8rem] border border-slate-600 hover:text-white hover:bg-slate-700"
          onClick={handleSubmit}
				>
					Upload Image
				</button>
			</form>
		</div>
	);
}

export default App;
