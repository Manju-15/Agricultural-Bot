let burger = document.querySelector(".burger");
let Links = document.querySelector(".nav-links");


burger.addEventListener('click',()=>{
    Links.classList.toggle("nav-show");
})