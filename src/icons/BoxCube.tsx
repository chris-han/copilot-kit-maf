import { Package as LucidePackage, LucideProps } from 'lucide-react';

const BoxCube = ({ className, ...props }: LucideProps) => {
  return <LucidePackage className={className} {...props} />;
};

export default BoxCube;